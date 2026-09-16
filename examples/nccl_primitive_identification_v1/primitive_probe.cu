// Native execution engine for the TRAF-94 source-faithful primitive study.
//
// JSON identity, frozen-request validation, and row construction live in
// probe.py.  This binary owns only CUDA/NCCL work: it constructs fresh
// communicators, binds the versioned source control block, executes warmups and
// iterations, verifies device output, and reports raw counters/timers.  Keeping
// JSON parsing out of CUDA code also makes the device boundary easy to audit.

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "traf94.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t status_ = (call);                                                \
    if (status_ != cudaSuccess)                                                  \
      throw std::runtime_error(std::string(#call) + ": " +                     \
                               cudaGetErrorString(status_));                     \
  } while (0)

#define NCCL_CHECK(call)                                                        \
  do {                                                                          \
    ncclResult_t status_ = (call);                                               \
    if (status_ != ncclSuccess)                                                  \
      throw std::runtime_error(std::string(#call) + ": " +                     \
                               ncclGetErrorString(status_));                     \
  } while (0)

#define DRIVER_CHECK(call)                                                      \
  do {                                                                          \
    CUresult status_ = (call);                                                   \
    if (status_ != CUDA_SUCCESS) {                                               \
      const char* text_ = nullptr;                                               \
      cuGetErrorString(status_, &text_);                                         \
      throw std::runtime_error(std::string(#call) + ": " +                     \
                               (text_ ? text_ : "unknown driver error"));        \
    }                                                                            \
  } while (0)

struct Options {
  std::string stage;
  std::string protocol;
  std::string family;
  std::string placement;
  std::string working_set;
  int ranks = 2;
  int channels = 1;
  int working_warps = 8;
  int available_sms = 0;  // zero means the full device.
  int reservations = 1;
  int warmups = 0;
  int iterations = 1;
  uint64_t useful_bytes = 16;
  uint64_t delay_cycles = 0;
  bool diagnostic = false;
};

static Options parse_options(int argc, char** argv) {
  std::map<std::string, std::string> values;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc || std::strncmp(argv[i], "--", 2) != 0)
      throw std::runtime_error("arguments must be --name value pairs");
    values.emplace(argv[i] + 2, argv[i + 1]);
  }
  auto required = [&](const char* name) -> const std::string& {
    auto it = values.find(name);
    if (it == values.end()) throw std::runtime_error(std::string("missing --") + name);
    return it->second;
  };
  auto optional = [&](const char* name, const char* fallback) -> std::string {
    auto it = values.find(name);
    return it == values.end() ? fallback : it->second;
  };

  Options o;
  o.stage = required("stage");
  o.protocol = required("protocol");
  o.family = required("family");
  o.placement = required("placement");
  o.working_set = optional("working-set", "reused");
  o.ranks = std::stoi(optional("ranks", "2"));
  o.channels = std::stoi(optional("channels", "1"));
  o.working_warps = std::stoi(optional("working-warps", "8"));
  o.available_sms = std::stoi(optional("available-sms", "0"));
  o.reservations = std::stoi(optional("reservations", "1"));
  o.warmups = std::stoi(optional("warmups", "0"));
  o.iterations = std::stoi(optional("iterations", "1"));
  o.useful_bytes = std::stoull(optional("useful-bytes", "16"));
  o.delay_cycles = std::stoull(optional("delay-cycles", "0"));
  o.diagnostic = std::stoi(optional("diagnostic", "0")) != 0;
  if (o.ranks < 2 || o.ranks > NCCL_TRAF94_MAX_RANKS || o.channels < 1 ||
      o.working_warps < 1 || o.reservations < 1 || o.iterations < 1 ||
      o.warmups < 0 || o.useful_bytes % sizeof(float) != 0)
    throw std::runtime_error("invalid numeric probe control");
  return o;
}

static int protocol_id(const std::string& value) {
  if (value == "LL") return 0;
  if (value == "LL128") return 1;
  if (value == "SIMPLE") return 2;
  throw std::runtime_error("unsupported protocol");
}

static int family_id(const std::string& value) {
  if (value == "already_ready") return ncclTraf94AlreadyReady;
  if (value == "delayed_publication" || value == "confirmation")
    return ncclTraf94DelayedPublication;
  if (value == "delayed_consumption") return ncclTraf94DelayedConsumption;
  if (value == "copy") return ncclTraf94Copy;
  if (value == "sum") return ncclTraf94Sum;
  throw std::runtime_error("unsupported family");
}

static int placement_id(const std::string& value) {
  if (value == "source_default") return ncclTraf94PlacementNA;
  if (value == "buffered") return ncclTraf94Buffered;
  if (value == "buffered_read") return ncclTraf94BufferedRead;
  throw std::runtime_error("unsupported placement");
}

static int popcount64(uint64_t value) { return __builtin_popcountll(value); }

// Only result fields are cleared between iterations.  Configuration is written
// again explicitly to make the reset boundary obvious and keep stale evidence
// from a warmup out of an ordinary sample.
static void reset_control(ncclTraf94Control* c, const Options& o) {
  std::memset(c, 0, sizeof(*c));
  c->magic = NCCL_TRAF94_MAGIC;
  c->enabled = 1;
  c->diagnostic = o.diagnostic ? 1 : 0;
  c->family = family_id(o.family);
  c->requestedProtocol = protocol_id(o.protocol);
  c->requestedPlacement = placement_id(o.placement);
  c->requestedWorkingWarps = o.working_warps;
  c->producerRank = 0;
  c->consumerRank = 1;
  c->forceEmpty = o.useful_bytes == 0 ? 1 : 0;
  c->delayCycles = o.delay_cycles;
  c->usefulBytes = o.useful_bytes;
  for (int r = 0; r < NCCL_TRAF94_MAX_RANKS; ++r)
    c->observedInputPtr[r] = UINT64_MAX;
  c->canary = 0x94c0ffee5a5aa5a5ULL;
}

struct Sample {
  double device_ns = 0;
  double event_us = 0;
  double wall_us = 0;
  uint64_t observed_delay = 0;
};

struct Probe {
  explicit Probe(const Options& options) : o(options) {
    p2p_intervention = o.stage == "ready_publication" || o.stage == "reuse";
    seed_direct_read_output = !p2p_intervention && o.protocol == "SIMPLE" &&
                              o.placement == "buffered_read";
    if (p2p_intervention && (o.ranks != 2 || o.channels != 1))
      throw std::runtime_error(
          "P2P intervention stages require exactly two ranks and one channel");
    DRIVER_CHECK(cuInit(0));
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices < o.ranks) throw std::runtime_error("insufficient visible GPUs");

    devs.resize(o.ranks);
    comms.resize(o.ranks);
    streams.resize(o.ranks);
    begins.resize(o.ranks);
    ends.resize(o.ranks);
    send.resize(o.ranks);
    recv.resize(o.ranks);
    registrations.resize(o.ranks, nullptr);
    granted_sms.resize(o.ranks);
    input_pointers.resize(o.ranks);
#if CUDA_VERSION >= 12040
    green.resize(o.ranks, nullptr);
#endif
    allocation_bytes = o.working_set == "rotating_64MiB" ? (64ULL << 20) :
                       std::max<uint64_t>(o.useful_bytes, sizeof(float));
    // Align rotating offsets and retain one full payload at the last slot.
    allocation_bytes = (allocation_bytes + 255) & ~uint64_t(255);

    for (int r = 0; r < o.ranks; ++r) devs[r] = r;
    NCCL_CHECK(ncclCommInitAll(comms.data(), o.ranks, devs.data()));
    CUDA_CHECK(cudaMallocManaged(&control, sizeof(*control), cudaMemAttachGlobal));
    // Set requested protocol/geometry before binding the control.  The patched
    // host scheduler snapshots these two fields to size the P2P kernel; each
    // iteration still resets all result fields below.
    reset_control(control, o);
    for (int r = 0; r < o.ranks; ++r) {
      cudaMemLocation location{};
      location.type = cudaMemLocationTypeDevice;
      location.id = r;
      CUDA_CHECK(cudaMemAdvise(control, sizeof(*control),
                               cudaMemAdviseSetAccessedBy, location));
      CUDA_CHECK(cudaSetDevice(r));
      cudaDeviceProp properties{};
      CUDA_CHECK(cudaGetDeviceProperties(&properties, r));
      granted_sms[r] = properties.multiProcessorCount;
      if (o.available_sms) {
#if CUDA_VERSION >= 12040
        CUcontext saved = nullptr;
        CUcontext converted = nullptr;
        CUdevResource all_sms{}, partition{}, remainder{}, check{};
        CUdevResourceDesc descriptor{};
        unsigned int groups = 1;
        DRIVER_CHECK(cuCtxGetCurrent(&saved));
        DRIVER_CHECK(cuDeviceGetDevResource(r, &all_sms, CU_DEV_RESOURCE_TYPE_SM));
        DRIVER_CHECK(cuDevSmResourceSplitByCount(
            &partition, &groups, &all_sms, &remainder, 0, o.available_sms));
        if (groups != 1)
          throw std::runtime_error("CUDA could not create one requested SM partition");
        DRIVER_CHECK(cuDevResourceGenerateDesc(&descriptor, &partition, 1));
        DRIVER_CHECK(cuGreenCtxCreate(&green[r], descriptor, r,
                                      CU_GREEN_CTX_DEFAULT_STREAM));
        DRIVER_CHECK(cuGreenCtxGetDevResource(green[r], &check,
                                              CU_DEV_RESOURCE_TYPE_SM));
        granted_sms[r] = check.sm.smCount;
        DRIVER_CHECK(cuCtxFromGreenCtx(&converted, green[r]));
        DRIVER_CHECK(cuCtxSetCurrent(converted));
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking));
        DRIVER_CHECK(cuCtxSetCurrent(saved));
#else
        throw std::runtime_error("CUDA toolkit lacks green-context support");
#endif
      } else {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking));
      }
      CUDA_CHECK(cudaEventCreate(&begins[r]));
      CUDA_CHECK(cudaEventCreate(&ends[r]));
      // NCCL's allocator creates exportable VMM handles.  Simple DirectRead
      // qualification depends on IPC registration succeeding; plain
      // cudaMalloc allocations on this CUDA 13 node are not exportable.
      NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void**>(&send[r]), allocation_bytes));
      NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void**>(&recv[r]), allocation_bytes));
      std::vector<float> input(allocation_bytes / sizeof(float), float(r + 1));
      // Ring DirectRead pulls from the registered output allocation once a
      // primitive has both recv and send roles.  Seed that allocation with the
      // local operand while retaining distinct buffers so both registrations
      // remain visible to NCCL's legacy Ring registration path.
      std::vector<float> output(allocation_bytes / sizeof(float),
                                seed_direct_read_output ? float(r + 1) : -1.0f);
      CUDA_CHECK(cudaMemcpy(send[r], input.data(), allocation_bytes, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(recv[r], output.data(), allocation_bytes, cudaMemcpyHostToDevice));
      if (o.protocol == "SIMPLE" && o.placement == "buffered_read") {
        // Registration is a request, not proof of DirectRead.  The source
        // remote-read/write counters provide that proof after execution.
        NCCL_CHECK(ncclCommRegister(comms[r], send[r], allocation_bytes,
                                    &registrations[r]));
        void* recv_handle = nullptr;
        NCCL_CHECK(ncclCommRegister(comms[r], recv[r], allocation_bytes,
                                    &recv_handle));
        recv_registrations.push_back(recv_handle);
      } else {
        recv_registrations.push_back(nullptr);
      }
      NCCL_CHECK(ncclTraf94SetControl(comms[r], control));
    }

  }

  ~Probe() {
    for (int r = 0; r < o.ranks; ++r) {
      cudaSetDevice(r);
      if (registrations[r]) ncclCommDeregister(comms[r], registrations[r]);
      if (recv_registrations[r]) ncclCommDeregister(comms[r], recv_registrations[r]);
      cudaEventDestroy(begins[r]);
      cudaEventDestroy(ends[r]);
      cudaStreamDestroy(streams[r]);
      ncclMemFree(send[r]);
      ncclMemFree(recv[r]);
      ncclCommDestroy(comms[r]);
    }
    if (control) cudaFree(control);
#if CUDA_VERSION >= 12040
    for (CUgreenCtx context : green)
      if (context) cuGreenCtxDestroy(context);
#endif
  }

  size_t offset_for(int iteration) const {
    if (o.working_set != "rotating_64MiB") return 0;
    uint64_t span = std::max<uint64_t>(o.useful_bytes, sizeof(float));
    uint64_t stride = (span + 255) & ~uint64_t(255);
    uint64_t slots = std::max<uint64_t>(1, allocation_bytes / stride);
    return (uint64_t(iteration) % slots) * stride / sizeof(float);
  }

  void sync_all() {
    for (int r = 0; r < o.ranks; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaStreamSynchronize(streams[r]));
    }
  }

  void issue_collective(size_t offset) {
    size_t count = std::max<uint64_t>(1, o.useful_bytes / sizeof(float));
    NCCL_CHECK(ncclGroupStart());
    int operations = o.family == "delayed_consumption" ? o.reservations : 1;
    for (int operation = 0; operation < operations; ++operation) {
      // Resource and confirmation cells define bytes per active channel.  One
      // independently addressable collective segment per requested channel
      // prevents NCCL's small-message tuner from collapsing the launch to a
      // smaller channel count while preserving the exact total useful bytes.
      int segments = (o.stage == "sharing" || o.stage == "confirmation")
                         ? o.channels
                         : 1;
      size_t segment_count = count / segments;
      for (int segment = 0; segment < segments; ++segment) {
        for (int r = 0; r < o.ranks; ++r) {
          CUDA_CHECK(cudaSetDevice(r));
          size_t segment_offset = offset + segment * segment_count;
          NCCL_CHECK(ncclAllReduce(send[r] + segment_offset,
                                   recv[r] + segment_offset, segment_count,
                                   ncclFloat, ncclSum, comms[r], streams[r]));
        }
      }
    }
    NCCL_CHECK(ncclGroupEnd());
  }

  void issue_p2p(size_t offset) {
    size_t count = std::max<uint64_t>(1, o.useful_bytes / sizeof(float));
    // Reservations remain ordered on the same producer/consumer streams so a
    // delayed consumer can keep the first FIFO slot live while later sends
    // exert backpressure.  Each matched pair gets its own NCCL plan; otherwise
    // SendRecv divides one block among the aggregated work items and the
    // requested primitive warp geometry is no longer realized.
    for (int i = 0; i < o.reservations; ++i) {
      NCCL_CHECK(ncclGroupStart());
      CUDA_CHECK(cudaSetDevice(0));
      NCCL_CHECK(ncclSend(send[0] + offset, count, ncclFloat, 1, comms[0], streams[0]));
      CUDA_CHECK(cudaSetDevice(1));
      NCCL_CHECK(ncclRecv(recv[1] + offset, count, ncclFloat, 0, comms[1], streams[1]));
      NCCL_CHECK(ncclGroupEnd());
    }
  }

  void issue(size_t offset) {
    if (p2p_intervention && o.family == "already_ready") {
      size_t count = std::max<uint64_t>(1, o.useful_bytes / sizeof(float));
      // NCCL requires both matching P2P APIs before launching.  The patched
      // receiver kernel gates primitive entry on the producer's exact source
      // publication witness, then starts the same-device interval.
      CUDA_CHECK(cudaSetDevice(1));
      CUDA_CHECK(cudaEventRecord(begins[1], streams[1]));
      ready_wall_begin = std::chrono::steady_clock::now();
      NCCL_CHECK(ncclGroupStart());
      CUDA_CHECK(cudaSetDevice(0));
      NCCL_CHECK(ncclSend(send[0] + offset, count, ncclFloat, 1,
                          comms[0], streams[0]));
      CUDA_CHECK(cudaSetDevice(1));
      NCCL_CHECK(ncclRecv(recv[1] + offset, count, ncclFloat, 0,
                          comms[1], streams[1]));
      NCCL_CHECK(ncclGroupEnd());
      CUDA_CHECK(cudaEventRecord(ends[1], streams[1]));
    } else if (p2p_intervention) {
      issue_p2p(offset);
    } else {
      issue_collective(offset);
    }
  }

  Sample once(int iteration) {
    sync_all();
    reset_control(control, o);
    size_t offset = offset_for(iteration);
    if (seed_direct_read_output) {
      // Ring DirectRead treats the registered output as this rank's local
      // operand once the primitive has both receive and send roles.  A prior
      // warmup leaves a reduced value there, so restore it from the untouched
      // input before every launch.  The copy is enqueued before all timing
      // events and before the primitive's source-local interval begins.
      size_t count = std::max<uint64_t>(1, o.useful_bytes / sizeof(float));
      for (int r = 0; r < o.ranks; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMemcpyAsync(recv[r] + offset, send[r] + offset,
                                   count * sizeof(float), cudaMemcpyDeviceToDevice,
                                   streams[r]));
      }
    }
    if (o.family != "already_ready") {
      for (int r = 0; r < o.ranks; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaEventRecord(begins[r], streams[r]));
      }
    }
    auto wall_begin = std::chrono::steady_clock::now();
    issue(offset);
    if (o.family != "already_ready") {
      for (int r = 0; r < o.ranks; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaEventRecord(ends[r], streams[r]));
      }
    }
    sync_all();
    auto wall_end = std::chrono::steady_clock::now();
    if (o.family == "already_ready") wall_begin = ready_wall_begin;
    for (int r = 0; r < o.ranks; ++r) {
      if (control->observedInputPtr[r] == UINT64_MAX)
        throw std::runtime_error("primitive did not record its input pointer");
      input_pointers[r].insert(control->observedInputPtr[r]);
    }

    Sample sample;
    int consumer = 1;
    float milliseconds = 0;
    CUDA_CHECK(cudaSetDevice(consumer));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begins[consumer], ends[consumer]));
    sample.event_us = milliseconds * 1000.0;
    sample.wall_us = std::chrono::duration<double, std::micro>(wall_end - wall_begin).count();
    if (control->intervalEnd[consumer] >= control->intervalBegin[consumer]) {
      // Primitive interval boundaries use Hopper's device-wide %globaltimer.
      // It is a nanosecond clock, unlike the SM-local clock64() used only to
      // implement and report the requested delay intervention.
      sample.device_ns =
          double(control->intervalEnd[consumer] - control->intervalBegin[consumer]);
    }
    sample.observed_delay = std::max(control->observedDelay[0], control->observedDelay[1]);
    return sample;
  }

  bool correctness() {
    size_t count = std::max<uint64_t>(1, o.useful_bytes / sizeof(float));
    size_t offset = offset_for(o.warmups + o.iterations - 1);
    std::vector<float> observed(count);
    int rank = (o.family == "already_ready" || o.family == "delayed_publication" ||
                o.family == "delayed_consumption") ? 1 : 0;
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaMemcpy(observed.data(), recv[rank] + offset,
                          count * sizeof(float), cudaMemcpyDeviceToHost));
    if (o.useful_bytes == 0 && seed_direct_read_output) {
      float expected = float(rank + 1);
      return std::all_of(observed.begin(), observed.end(),
                         [&](float x) { return x == expected; });
    }
    if (o.useful_bytes == 0)
      return std::all_of(observed.begin(), observed.end(),
                         [](float x) { return x == -1.0f; });
    if (o.family == "sum" || o.family == "confirmation" ||
        (!p2p_intervention && (o.family == "already_ready" ||
                               o.family == "delayed_publication" ||
                               o.family == "delayed_consumption"))) {
      float expected = float(o.ranks * (o.ranks + 1) / 2);
      for (size_t i = 0; i < observed.size(); ++i) {
        if (observed[i] != expected) {
          // This diagnostic is outside every timing interval and only appears
          // on a failed run.  It makes a retained correctness failure useful
          // without dumping the payload or weakening exact comparison.
          std::cerr << "correctness mismatch at element " << i << ": got "
                    << observed[i] << ", expected " << expected << '\n';
          return false;
        }
      }
      return true;
    }
    if (o.family == "copy") {
      // The source hook selects one loaded operand at every reduction site.
      // Ring order decides which rank reaches the final output, but the known
      // legal set is exactly the initialized rank values.
      return std::all_of(observed.begin(), observed.end(), [&](float x) {
        return std::isfinite(x) && x >= 1.0f && x <= float(o.ranks) &&
               x == std::floor(x);
      });
    }
    if (p2p_intervention) {
      // Readiness/reuse interventions are one-way rank-0 transfers.  They
      // deliberately avoid reduction work so their interval isolates the
      // protocol state machine.
      return std::all_of(observed.begin(), observed.end(),
                         [](float x) { return x == 1.0f; });
    }
    return std::all_of(observed.begin(), observed.end(),
                       [](float x) { return x == 1.0f; });
  }

  const Options& o;
  bool p2p_intervention = false;
  bool seed_direct_read_output = false;
  std::vector<int> devs;
  std::vector<ncclComm_t> comms;
  std::vector<cudaStream_t> streams;
  std::vector<cudaEvent_t> begins, ends;
  std::vector<float*> send, recv;
  std::vector<void*> registrations, recv_registrations;
  std::vector<int> granted_sms;
  std::vector<std::set<uint64_t>> input_pointers;
#if CUDA_VERSION >= 12040
  std::vector<CUgreenCtx> green;
#endif
  ncclTraf94Control* control = nullptr;
  uint64_t allocation_bytes = 0;
  std::chrono::steady_clock::time_point ready_wall_begin;
};

template <typename T>
static void print_array(const std::vector<T>& values) {
  std::cout << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << values[i];
  }
  std::cout << ']';
}

static void print_peer_matrix(
    const uint32_t values[NCCL_TRAF94_MAX_RANKS][NCCL_TRAF94_MAX_CHANNELS],
    int ranks) {
  std::cout << '[';
  for (int r = 0; r < ranks; ++r) {
    if (r) std::cout << ',';
    std::cout << '[';
    for (int c = 0; c < NCCL_TRAF94_MAX_CHANNELS; ++c) {
      if (c) std::cout << ',';
      // Device evidence stores rank+1 so zero can mean no peer.  Decode here
      // once and expose the conventional -1 sentinel to the strict adapter.
      std::cout << (values[r][c] == 0 ? -1 : int(values[r][c] - 1));
    }
    std::cout << ']';
  }
  std::cout << ']';
}

int main(int argc, char** argv) {
  try {
    Options options = parse_options(argc, argv);
    Probe probe(options);
    for (int i = 0; i < options.warmups; ++i) probe.once(i);
    std::vector<double> device_ns, events, walls;
    std::vector<uint64_t> delays;
    for (int i = 0; i < options.iterations; ++i) {
      Sample sample = probe.once(options.warmups + i);
      device_ns.push_back(sample.device_ns);
      events.push_back(sample.event_us);
      walls.push_back(sample.wall_us);
      delays.push_back(sample.observed_delay);
    }
    ncclTraf94Control* c = probe.control;
    bool correct = probe.correctness();
    std::cout << "{\"schema\":\"simllm-nccl-primitive-native-v1\","
              << "\"correctness\":" << (correct ? "true" : "false") << ','
              << "\"canary_ok\":"
              << (c->canary == 0x94c0ffee5a5aa5a5ULL ? "true" : "false") << ','
              << "\"unsupported_site\":" << c->unsupportedSite << ','
              << "\"samples\":{\"same_device_interval\":";
    print_array(device_ns);
    std::cout << ",\"cuda_event\":";
    print_array(events);
    std::cout << ",\"host_wall\":";
    print_array(walls);
    std::cout << ",\"observed_local_delay\":";
    print_array(delays);
    std::cout << "},\"counts\":{"
              << "\"readiness_checks\":" << c->readinessChecks << ','
              << "\"unsuccessful_checks\":" << c->unsuccessfulChecks << ','
              << "\"successful_observations\":" << c->successfulObservations << ','
              << "\"tail_publications\":" << c->tailPublications << ','
              << "\"head_publications\":" << c->headPublications << ','
              << "\"source_load_bytes\":" << c->sourceLoadBytes << ','
              << "\"shared_staging_bytes\":" << c->sharedStagingBytes << ','
              << "\"remote_read_bytes\":" << c->remoteReadBytes << ','
              << "\"remote_write_bytes\":" << c->remoteWriteBytes << ','
              << "\"consumed_steps\":" << c->consumedSteps << "},"
              << "\"realized\":{\"allocation_bytes\":" << probe.allocation_bytes
              << ",\"protocols\":";
    std::vector<uint32_t> protocols(c->realizedProtocol,
                                    c->realizedProtocol + options.ranks);
    print_array(protocols);
    std::cout << ",\"block_threads\":";
    std::vector<uint32_t> blocks(c->realizedBlockThreads,
                                 c->realizedBlockThreads + options.ranks);
    print_array(blocks);
    std::cout << ",\"working_warps\":";
    std::vector<uint32_t> warps(c->realizedWorkWarps,
                                c->realizedWorkWarps + options.ranks);
    print_array(warps);
    std::cout << ",\"channel_masks\":";
    std::vector<uint32_t> channels(c->realizedChannelMask,
                                   c->realizedChannelMask + options.ranks);
    print_array(channels);
    std::cout << ",\"placement_masks\":";
    std::vector<uint32_t> placements(c->realizedPlacementMask,
                                     c->realizedPlacementMask + options.ranks);
    print_array(placements);
    std::cout << ",\"recv_peers\":";
    print_peer_matrix(c->realizedRecvPeer, options.ranks);
    std::cout << ",\"send_peers\":";
    print_peer_matrix(c->realizedSendPeer, options.ranks);
    std::cout << ",\"active_channels\":[";
    for (int r = 0; r < options.ranks; ++r) {
      if (r) std::cout << ',';
      std::cout << __builtin_popcount(c->realizedChannelMask[r]);
    }
    std::cout << "],\"resident_sms\":[";
    for (int r = 0; r < options.ranks; ++r) {
      if (r) std::cout << ',';
      std::cout << popcount64(c->realizedSmMaskLo[r]) +
                       popcount64(c->realizedSmMaskHi[r]) +
                       popcount64(c->realizedSmMask2[r]);
    }
    std::cout << "],\"granted_sms\":";
    print_array(probe.granted_sms);
    std::cout << ",\"input_pointer_distinct\":[";
    for (int r = 0; r < options.ranks; ++r) {
      if (r) std::cout << ',';
      std::cout << probe.input_pointers[r].size();
    }
    std::cout << "],\"input_pointer_span_bytes\":[";
    for (int r = 0; r < options.ranks; ++r) {
      if (r) std::cout << ',';
      const auto& pointers = probe.input_pointers[r];
      std::cout << (*pointers.rbegin() - *pointers.begin());
    }
    std::cout << "]}}\n";
    return correct ? 0 : 5;
  } catch (const std::exception& error) {
    std::cerr << "primitive_probe: " << error.what() << '\n';
    return 2;
  }
}

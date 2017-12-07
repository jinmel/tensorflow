#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "gpu_record_yielder.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
REGISTER_OP("GpuAwareRecordInput")
    .Output("records: string")
    .Attr("file_pattern: string")
    .Attr("file_random_seed: int = 301")
    .Attr("file_shuffle_shift_ratio: float = 0")
    .Attr("file_buffer_size: int = 10000")
    .Attr("file_parallelism: int = 16")
    .Attr("batch_size: int = 32")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Emits randomized records.

records: A tensor of shape [batch_size].
file_pattern: Glob pattern for the data files.
file_random_seed: Random seeds used to produce randomized records.
file_shuffle_shift_ratio: Shifts the list of files after the list is randomly
    shuffled.
file_buffer_size: The randomization shuffling buffer.
file_parallelism: How many sstables are opened and concurrently iterated over.
batch_size: The batch size.
)doc");
class GpuAwareRecordInputOp : public OpKernel {
 public:
  explicit GpuAwareRecordInputOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
#define GETATTR(TYPE, FIELD) \
  TYPE FIELD;                \
  OP_REQUIRES_OK(ctx, ctx->GetAttr(#FIELD, &FIELD));

    GETATTR(string, file_pattern);
    GETATTR(int64, file_random_seed);
    GETATTR(float, file_shuffle_shift_ratio);
    GETATTR(int64, file_buffer_size);
    GETATTR(int64, file_parallelism);
    GETATTR(int64, batch_size);
#undef GETATTR

    RecordYielder::Options yopts;
    yopts.file_pattern = file_pattern;
    yopts.seed = file_random_seed;
    yopts.bufsize = file_buffer_size;
    yopts.file_shuffle_shift_ratio = file_shuffle_shift_ratio;
    yopts.parallelism = file_parallelism;
    yielder_ = std::unique_ptr<RecordYielder>(new RecordYielder(ctx, yopts));

    batch_size_ = batch_size;
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor out(DT_STRING, {batch_size_});
    auto t_out = out.flat<string>();
    for (int i = 0; i < batch_size_; ++i) {
      OP_REQUIRES_OK(ctx, yielder_->YieldOne(&t_out(i)));
    }
    ctx->set_output(0, out);
  }

 private:
  int64 batch_size_;
  std::unique_ptr<RecordYielder> yielder_;
};

REGISTER_KERNEL_BUILDER(Name("GpuAwareRecordInput").Device(DEVICE_CPU),
                        GpuAwareRecordInputOp);
}  // namespace tensorflow

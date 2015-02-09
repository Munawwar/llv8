#include "low-chunk.h"

namespace v8 {
namespace internal {

LowChunk::LowChunk(CompilationInfo* info, HGraph* graph)
    : info_(info),
      graph_(graph) {}


}  // namespace v8
}  // namespace internal

// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_LOWCHUNK_H_
#define V8_LOWCHUNK_H_

//#include "compiler.h"
#include "hydrogen.h"
#include "zone-allocator.h"


namespace v8 {
namespace internal {

class LowChunk : public ZoneObject {
  public:
    virtual ~LowChunk();
    //virtual LowChunk* NewChunk(HGraph *graph) = 0;
    virtual Handle<Code> Codegen() = 0;
    Zone* zone() const { return info_->zone(); }

    CompilationInfo* info() const { return info_; }
    HGraph* graph() const { return graph_; }
    Isolate* isolate() const;

    void AddStabilityDependency(Handle<Map> map) {
      DCHECK(map->is_stable());
      if (!map->CanTransition()) return;
      DCHECK(!info()->IsStub());
      stability_dependencies_.insert(map);
    }

    void AddDeprecationDependency(Handle<Map> map) {
      DCHECK(map->is_deprecated());
      if (!map->CanBeDeprecated()) return;
      DCHECK(!info()->IsStub());
      deprecation_dependencies_.insert(map);
    }

  protected:
    using MapLess = std::less<Handle<Map> >;
    using MapAllocator = zone_allocator<Handle<Map> >;
    using MapSet = std::set<Handle<Map>, MapLess, MapAllocator>;

    LowChunk(CompilationInfo* info, HGraph* graph);

    MapSet stability_dependencies_;
    MapSet deprecation_dependencies_;

  private:
    CompilationInfo* info_;
    HGraph* const graph_;
};

class LowChunkBuilderBase BASE_EMBEDDED {
  public:
    virtual ~LowChunkBuilderBase() {} // FIXME(llvm): virtuality now seems redundant
    explicit LowChunkBuilderBase(CompilationInfo* info, HGraph* graph);

    void Abort(BailoutReason reason);
    void Retry(BailoutReason reason);

  protected:
    enum Status { UNUSED, BUILDING, DONE, ABORTED };

    LowChunk* chunk() const { return chunk_; }
    CompilationInfo* info() const { return info_; }
    HGraph* graph() const { return graph_; }
    Isolate* isolate() const;
    Heap* heap() const { return isolate()->heap(); }
    Zone* zone() const { return zone_; }
    int argument_count() const { return argument_count_; }

    bool is_unused() const { return status_ == UNUSED; }
    bool is_building() const { return status_ == BUILDING; }
    bool is_done() const { return status_ == DONE; }
    bool is_aborted() const { return status_ == ABORTED; }

    LowChunk* chunk_;
    CompilationInfo* info_;
    HGraph* const graph_;
    Status status_;
    int argument_count_;
    HBasicBlock* current_block_;
    HBasicBlock* next_block_;

  private:
   Zone* zone_;
};

// FIXME(llvm): it seems we don't use this class at all
// (it has only 1 subclass).
class LowCodeGenBase BASE_EMBEDDED {
  public:
    LowCodeGenBase(LowChunk* chunk, CompilationInfo* info)
        : chunk_(chunk),
          info_(info),
          zone_(info->zone()) {}
    virtual ~LowCodeGenBase() {}

    LowChunk* chunk() const { return chunk_; }
    HGraph* graph() const { return chunk()->graph(); }
    Zone* zone() const { return zone_; }
    CompilationInfo* info() const { return info_; }
    Isolate* isolate() const { return info_->isolate(); }
    Factory* factory() const { return isolate()->factory(); }
    Heap* heap() const { return isolate()->heap(); }

    // Try to generate native code for the entire chunk, but it may fail if the
    // chunk contains constructs we cannot handle. Returns true if the
    // code generation attempt succeeded.
    // FIXME(llvm): return this method to the child class (make non-virtual)
    virtual bool GenerateCode() = 0;

    // Finish the code by setting stack height, safepoint, and bailout
    // information on it.
    // FIXME(llvm): return this method to the child class (make non-virtual)
    // Or use it...
    virtual void FinishCode(Handle<Code> code) = 0;
  protected:
    LowChunk* const chunk_;
    CompilationInfo* const info_;
    Zone* zone_;
};

} }  // namespace v8::internal

#endif  // V8_LOWCHUNK_H_

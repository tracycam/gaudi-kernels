// SPDX-License-Identifier: Apache-2.0
// V1: owned, single-stream, static-bucket native graph. No torch/c10 types.
#ifndef GK_NATIVE_GRAPH_H
#define GK_NATIVE_GRAPH_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define GKG_ABI_VERSION 1
typedef struct gkg_graph gkg_graph;
typedef enum {
 GKG_OK=0, GKG_INVALID=1, GKG_STATE=2, GKG_RANGE=3, GKG_UNINITIALIZED=4,
 GKG_READ_ONLY=5, GKG_UNSUPPORTED=6, GKG_RUNTIME=7, GKG_BUSY=8, GKG_STALE=9
} gkg_status;
typedef enum {
 GKG_INPUT=1, GKG_OUTPUT=2, GKG_TEMP=3,
 GKG_EXTERNAL_READ=4, GKG_EXTERNAL_RW=5
} gkg_role;
typedef enum { GKG_F32=1, GKG_BF16=2, GKG_I32=3, GKG_U8=4 } gkg_dtype;
typedef enum { GKG_ALL_REDUCE=1, GKG_ALL_GATHER=2, GKG_REDUCE_SCATTER=3 } gkg_collective;
typedef struct {
 uint32_t size, version, device, flights;
 const char* artifact_directory;
} gkg_config;
typedef struct {
 uint32_t size, version;
 const char* tensor;
 const char* buffer;
 uint64_t offset;
} gkg_binding;
typedef struct {
 uint32_t size, version;
 const char* name;
 const void* data;
 uint64_t bytes;
} gkg_input;
gkg_status gkg_create(const gkg_config*,gkg_graph**);
// External addresses are borrowed; caller must retain and freeze their owners.
// INPUT/OUTPUT/TEMP addresses identify capture storage, never replay storage.
gkg_status gkg_buffer(gkg_graph*,const char*,uint64_t capture_address,uint64_t bytes,gkg_role);
// Declare/promote the exact range of an automatically captured intermediate.
gkg_status gkg_output(gkg_graph*,const char*,uint64_t capture_address,uint64_t bytes);
// Clones via public recipe serialization; producer can then destroy its recipe.
gkg_status gkg_recipe(gkg_graph*,const char*,void* synapse_recipe);
gkg_status gkg_recipe_load(gkg_graph*,const char*,const char* recipe_file);
gkg_status gkg_launch(gkg_graph*,const char* recipe,const gkg_binding*,uint32_t count);
gkg_status gkg_comm(gkg_graph*,const char*,void* hccl_communicator);
// Required when hcclCommSynDevice does not populate its output on this SDK.
// Caller explicitly attests communicator ownership on the acquired device.
gkg_status gkg_comm_on_device(gkg_graph*,const char*,void* hccl_communicator,uint32_t device);
gkg_status gkg_collect(gkg_graph*,gkg_collective,const char* comm,const char* src,uint64_t src_offset,
                      const char* dst,uint64_t dst_offset,uint64_t count,gkg_dtype);
gkg_status gkg_copy(gkg_graph*,const char* src,uint64_t src_offset,const char* dst,uint64_t dst_offset,uint64_t bytes);
// A framework may reserve all HBM in its allocator. Supply a dedicated arena
// from that allocator instead of allocating it again through Synapse. The
// caller retains this arena's owner until graph destruction; it must not alias
// any capture/external buffer. No framework type enters this C ABI.
gkg_status gkg_pool_requirements(gkg_graph*,uint64_t* bytes);
gkg_status gkg_bind_pool(gkg_graph*,uint64_t address,uint64_t bytes);
gkg_status gkg_instantiate(gkg_graph*);
// Inputs copied to a free flight's owned pinned storage. Enqueues and returns;
// no synchronize. Device buffers/workspace are reused in one stream's order.
gkg_status gkg_replay(gkg_graph*,const gkg_input*,uint32_t count,uint64_t* ticket);
gkg_status gkg_query(gkg_graph*,uint64_t ticket);
gkg_status gkg_wait(gkg_graph*,uint64_t ticket);
gkg_status gkg_read(gkg_graph*,uint64_t ticket,const char* output,void* dst,uint64_t bytes);
gkg_status gkg_release(gkg_graph*,uint64_t ticket);
gkg_status gkg_check_external(gkg_graph*,const char*,uint64_t address,uint64_t bytes);
gkg_status gkg_address(gkg_graph*,const char*,uint64_t* address);
gkg_status gkg_stream(gkg_graph*,void** synapse_stream);
gkg_status gkg_report(gkg_graph*,const char* file);
// Optional LD_PRELOAD public-API producer (libgkg_capture.so). Single host
// submission thread and stream; host I/O, synchronization and allocation are
// forbidden during recording. End returns a failure instead of a partial plan.
gkg_status gkg_capture_begin(gkg_graph*,void* synapse_stream);
gkg_status gkg_capture_end(gkg_graph*,const char* coverage_report);
const char* gkg_error(gkg_graph*);
// Returns BUSY instead of freeing resources referenced by an outstanding ticket.
gkg_status gkg_destroy(gkg_graph*);
#ifdef __cplusplus
}
#endif
#endif

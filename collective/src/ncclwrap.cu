#include "ncclwrap.h"
#include "common.h"
#include <pthread.h>
#include <stdio.h>

namespace paddle {
namespace communication {
namespace dgc{

static enum { ncclUninitialized, ncclInitializing, ncclInitialized, ncclError } ncclState = ncclUninitialized;

/*Function Pointers*/
static ncclResult_t (*ncclCommCountFuncPoint)(const ncclComm_t comm, int* count);
static ncclResult_t (*ncclAllGatherFuncPoint)(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

bool warpNcclSymbols(void) {
  if (ncclState == ncclInitialized) {
    return true;
  } else if (ncclState == ncclError) {
    return false;
  }

  if (__sync_bool_compare_and_swap(&ncclState, ncclUninitialized, ncclInitializing) == false) {
    // Another thread raced in front of us. Wait for it to be done.
    while (ncclState == ncclInitializing) pthread_yield();
    return (ncclState == ncclInitialized) ? true : false;
  }

  static void* ncclhandle = NULL;
  void* tmp = NULL;
  void** cast = NULL;

  ncclhandle=dlopen("libnccl.so", RTLD_NOW);
  if (!ncclhandle) {
    ncclhandle=dlopen("libnccl.so.2", RTLD_NOW);
    if (!ncclhandle) {
      LOGERR("Failed to open libnccl.so[.2]");
      goto teardown;
    }
  }

#define LOAD_SYM(handle, symbol, funcptr) do {                 \
    cast = (void**)&funcptr;                                   \
    tmp = dlsym(handle, symbol);                               \
    if (tmp == NULL) {                                         \
      LOGERR("dlsym failed on %s - %s\n", symbol, dlerror());  \
      goto teardown;                                           \
    }                                                          \
    *cast = tmp;                                               \
  } while (0)

  LOAD_SYM(ncclhandle, "ncclCommCount", ncclCommCountFuncPoint);
  LOAD_SYM(ncclhandle, "ncclAllGather", ncclAllGatherFuncPoint);

  ncclState = ncclInitialized;
  return true;

teardown:
  ncclCommCountFuncPoint = NULL;
  ncclAllGatherFuncPoint = NULL;

  if (ncclhandle != NULL) dlclose(ncclhandle);
  ncclState = ncclError;
  return false;
}

#define NCCL_CHECK(cmd) do {                        \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    LOGERR("Failed, NCCL error '%s'",               \
           ncclGetErrorString(r));                  \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)
 
bool warpNcclCommCount(const ncclComm_t comm, int* count) {
  if (ncclCommCountFuncPoint == NULL) {
    LOGERR("lib nccl not initialized.");
    exit(EXIT_FAILURE);
    return false;
  }
  NCCL_CHECK(ncclCommCountFuncPoint(comm, count));
  return true;
}

bool warpNcclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  if (ncclAllGatherFuncPoint == NULL) {
    LOGERR("lib nccl not initialized.");
    exit(EXIT_FAILURE);
    return false;
  }
  NCCL_CHECK(ncclAllGatherFuncPoint(sendbuff, recvbuff, sendcount, datatype, comm, stream));
  return true;
}

}  // namespace dgc
}  // namespace communication
}  // namespace paddle

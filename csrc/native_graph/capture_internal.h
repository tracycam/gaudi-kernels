#pragma once
#include "graph.h"
extern "C" gkg_status gkg_capture_address(gkg_graph*,uint64_t,uint64_t,int,char*,uint32_t,uint64_t*);
extern "C" gkg_status gkg_capture_communicator(gkg_graph*,void*,char*,uint32_t);
extern "C" gkg_status gkg_capture_fail(gkg_graph*,gkg_status,const char*);

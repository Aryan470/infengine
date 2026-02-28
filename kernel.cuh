#pragma once
#include "manager.h"

int initialize_request_context(RequestContext* context);
int prefill(RequestContext* context);
int decode(RequestContext* context);
int cleanup_request_context(RequestContext* context);
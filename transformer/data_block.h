#ifndef __DATA_BLOCK_H__
#define __DATA_BLOCK_H__

#include "haipin_define.h"

void new_fd(ForwardData* in_fd, int batch_num, int height, int width);

void fd_assignment(ForwardData* in_fd, ForwardData* out_fd);

void delete_fd(ForwardData* in_fd);
#endif

###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXS library.                                     #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxs/                          #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################

CC ?= gcc
AR ?= ar
CFLAGS ?= -O2 -fPIC -Wall -Wextra -std=c99 -pedantic
CPPFLAGS ?= -Iinclude

BUILD_DIR ?= obj
LIB_DIR ?= lib
TARGET ?= $(LIB_DIR)/libxs.a

# All current C sources in src/
SRC := $(wildcard src/*.c)
OBJ := $(patsubst src/%.c,$(BUILD_DIR)/%.o,$(SRC))

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ) | $(LIB_DIR)
	$(AR) rcs $@ $^

$(BUILD_DIR)/%.o: src/%.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR) $(LIB_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

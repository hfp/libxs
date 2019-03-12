/******************************************************************************
** Copyright (c) 2018-2019, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
#if defined(__PYTHON) && defined(LIBXS_BUILD) && !defined(__STATIC)
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <Python.h> /* must be included first */
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif
#endif
#include <libxs.h>


#if defined(__PYTHON) && defined(LIBXS_BUILD) && !defined(__STATIC)

LIBXS_API PyObject* libxspy_get_target_arch(PyObject* self, PyObject* args);
LIBXS_API PyObject* libxspy_get_target_arch(PyObject* self, PyObject* args)
{
  LIBXS_UNUSED(self); LIBXS_UNUSED(args);
  return PyString_InternFromString(libxs_get_target_arch());
}

LIBXS_API PyObject* libxspy_set_target_arch(PyObject* self, PyObject* args);
LIBXS_API PyObject* libxspy_set_target_arch(PyObject* self, PyObject* args)
{
  int ivalue = LIBXS_TARGET_ARCH_UNKNOWN;
  char* svalue = NULL;
  LIBXS_UNUSED(self);
  if (0 != PyArg_ParseTuple(args, "s", &svalue)) {
    libxs_set_target_arch(svalue);
  }
  else if (0 != PyArg_ParseTuple(args, "i", &ivalue)) {
    libxs_set_target_archid(ivalue);
  }
  else { /* error */
    return NULL;
  }
  Py_RETURN_NONE;
}


LIBXS_API PyObject* libxspy_get_target_archid(PyObject* self, PyObject* args);
LIBXS_API PyObject* libxspy_get_target_archid(PyObject* self, PyObject* args)
{
  LIBXS_UNUSED(self); LIBXS_UNUSED(args);
  return Py_BuildValue("i", libxs_get_target_archid());
}

LIBXS_API PyObject* libxspy_set_target_archid(PyObject* self, PyObject* args);
LIBXS_API PyObject* libxspy_set_target_archid(PyObject* self, PyObject* args)
{
  int value = LIBXS_TARGET_ARCH_UNKNOWN;
  LIBXS_UNUSED(self);
  if (0 != PyArg_ParseTuple(args, "i", &value)) {
    libxs_set_target_archid(value);
  }
  else { /* error */
    return NULL;
  }
  Py_RETURN_NONE;
}


LIBXS_API PyObject* libxspy_get_verbosity(PyObject* self, PyObject* args);
LIBXS_API PyObject* libxspy_get_verbosity(PyObject* self, PyObject* args)
{
  LIBXS_UNUSED(self); LIBXS_UNUSED(args);
  return Py_BuildValue("i", libxs_get_verbosity());
}

LIBXS_API PyObject* libxspy_set_verbosity(PyObject* self, PyObject* args);
LIBXS_API PyObject* libxspy_set_verbosity(PyObject* self, PyObject* args)
{
  int value = 0;
  LIBXS_UNUSED(self);
  if (0 != PyArg_ParseTuple(args, "i", &value)) {
    libxs_set_verbosity(value);
  }
  else { /* error */
    return NULL;
  }
  Py_RETURN_NONE;
}


PyMODINIT_FUNC initlibxs(void);
PyMODINIT_FUNC initlibxs(void)
{
  static PyMethodDef pymethod_def[] = {
    { "GetTargetArch", libxspy_get_target_arch, METH_NOARGS,
      PyDoc_STR("Get the name of the code path.") },
    { "SetTargetArch", libxspy_set_target_arch, METH_VARARGS,
      PyDoc_STR("Set the name of the code path.") },
    { "GetTargetArchId", libxspy_get_target_archid, METH_NOARGS,
      PyDoc_STR("Get the id of the code path.") },
    { "SetTargetArchId", libxspy_set_target_archid, METH_VARARGS,
      PyDoc_STR("Set the id of the code path.") },
    { "GetVerbosity", libxspy_get_verbosity, METH_NOARGS,
      PyDoc_STR("Get the verbosity level.") },
    { "SetVerbosity", libxspy_set_verbosity, METH_VARARGS,
      PyDoc_STR("Set the verbosity level.") },
    { NULL, NULL, 0, NULL } /* end of table */
  };
  PyObject *const pymod = Py_InitModule3("libxs", pymethod_def, PyDoc_STR(
    "Library targeting Intel Architecture for small, dense or "
    "sparse matrix multiplications, and small convolutions."));
  PyModule_AddIntConstant(pymod, "VERSION_API", LIBXS_VERSION2(LIBXS_VERSION_MAJOR, LIBXS_VERSION_MINOR));
  PyModule_AddIntConstant(pymod, "VERSION_ALL", LIBXS_VERSION4(LIBXS_VERSION_MAJOR, LIBXS_VERSION_MINOR,
                                                                LIBXS_VERSION_UPDATE, LIBXS_VERSION_PATCH));
  PyModule_AddIntConstant(pymod, "VERSION_MAJOR", LIBXS_VERSION_MAJOR);
  PyModule_AddIntConstant(pymod, "VERSION_MINOR", LIBXS_VERSION_MINOR);
  PyModule_AddIntConstant(pymod, "VERSION_UPDATE", LIBXS_VERSION_UPDATE);
  PyModule_AddIntConstant(pymod, "VERSION_PATCH", LIBXS_VERSION_PATCH);
  PyModule_AddStringConstant(pymod, "VERSION", LIBXS_VERSION);
  PyModule_AddStringConstant(pymod, "BRANCH", LIBXS_BRANCH);
  PyModule_AddIntConstant(pymod, "TARGET_ARCH_UNKNOWN", LIBXS_TARGET_ARCH_UNKNOWN);
  PyModule_AddIntConstant(pymod, "TARGET_ARCH_GENERIC", LIBXS_TARGET_ARCH_GENERIC);
  PyModule_AddIntConstant(pymod, "X86_IMCI", LIBXS_X86_IMCI);
  PyModule_AddIntConstant(pymod, "X86_GENERIC", LIBXS_X86_GENERIC);
  PyModule_AddIntConstant(pymod, "X86_SSE3", LIBXS_X86_SSE3);
  PyModule_AddIntConstant(pymod, "X86_SSE4", LIBXS_X86_SSE4);
  PyModule_AddIntConstant(pymod, "X86_AVX", LIBXS_X86_AVX);
  PyModule_AddIntConstant(pymod, "X86_AVX2", LIBXS_X86_AVX2);
  PyModule_AddIntConstant(pymod, "X86_AVX512", LIBXS_X86_AVX512);
  PyModule_AddIntConstant(pymod, "X86_AVX512_MIC", LIBXS_X86_AVX512_MIC);
  PyModule_AddIntConstant(pymod, "X86_AVX512_KNM", LIBXS_X86_AVX512_KNM);
  PyModule_AddIntConstant(pymod, "X86_AVX512_CORE", LIBXS_X86_AVX512_CORE);
  PyModule_AddIntConstant(pymod, "X86_AVX512_CLX", LIBXS_X86_AVX512_CLX);
  libxs_init(); /* initialize LIBXS */
}

#endif /*defined(__PYTHON) && defined(LIBXS_BUILD) && !defined(__STATIC)*/

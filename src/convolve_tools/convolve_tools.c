/*  Example of wrapping the cos function from math.h using the Numpy-C-API. */

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

#define QUOTE(s) # s
#define get_double(a, i, j) *((double *) PyArray_GETPTR2(a, i, j))
#define set_double(a, i, j, value) *((double *) PyArray_GETPTR2(a, i, j)) = value;

#define get_int(a, i, j) *((int *) PyArray_GETPTR2(a, i, j))
#define set_int(a, i, j, value) *((int *) PyArray_GETPTR2(a, i, j)) = value;

/* check that the dimensions of an array match the expected value */
int check_dims(PyArrayObject* a, int expected_ndim)
{
    if (PyArray_NDIM(a) != expected_ndim) {
        PyErr_Format(PyExc_ValueError,
                     "%s array is %d-dimensional, but expected to be %d-dimensional",
                     QUOTE(a), PyArray_NDIM(a), expected_ndim);
        return -1;
    }

    return 0;
}

/* Check the the dimensions of two arrays are eqaul to one another */
int check_dims_equal(PyArrayObject* a, PyArrayObject* b)
{
    if (PyArray_NDIM(a) != PyArray_NDIM(b)) {
        PyErr_Format(PyExc_ValueError,
                     "%s array is %d-dimensional, but %s array is %d-dimensional",
                     QUOTE(a), PyArray_NDIM(a), QUOTE(b), PyArray_NDIM(b));
        return -1;
    }

    return 0;
}

void convolve(PyArrayObject* image, PyArrayObject* mask, PyArrayObject* kernel, PyObject* out)
{
    int rows = PyArray_DIM(image, 0);
    int cols = PyArray_DIM(image, 1);
    npy_intp kernelSize = PyArray_DIM(kernel, 0);

    int kRows = kernelSize;
    int kCols = kernelSize;

    // find center position of kernel (half of kernel size)
    int kCenterX = kCols / 2;
    int kCenterY = kRows / 2;

    for(int i=0; i < rows; ++i)              // rows
    {
        for(int j=0; j < cols; ++j)          // columns
        {
            for(int m=0; m < kRows; ++m)     // kernel rows
            {
                int mm = kRows - 1 - m;      // row index of flipped kernel

                for(int n=0; n < kCols; ++n) // kernel columns
                {
                    int nn = kCols - 1 - n;  // column index of flipped kernel

                    // index of input signal, used for checking boundary
                    int ii = i + (m - kCenterY);
                    int jj = j + (n - kCenterX);

                    // ignore input samples which are out of bound
                    if(ii >= 0 && ii < rows && jj >= 0 && jj < cols )
                    {
                        int mask_value = get_int(mask, i,j);
                        if (mask_value != 0) {
                            double value = get_double(out, i, j);
                            value += get_double(image,ii,jj) * get_double(kernel, mm, nn) * mask_value;
                            set_double(out, i, j, value);
                            // out[i][j] += in[ii][jj] * kernel[mm][nn];
                        }
                    }
                }
            }
        }
    }
}

static PyObject* deformable_covolution(PyObject* self, PyObject* args)
{
    PyArrayObject *image;
    PyArrayObject *mask;
    PyArrayObject *kernel;

    PyObject      *out_array;

    NpyIter *in_iter;
    NpyIter *out_iter;
    NpyIter_IterNextFunc *in_iternext;
    NpyIter_IterNextFunc *out_iternext;

    /*  parse the image, mask and kernel */
    if (!PyArg_ParseTuple(args, "OOO", &image, &mask, &kernel))
        return NULL;

    // if (!check_dims(image, 2)) return NULL;
    // if (!check_dims(mask, 2)) return NULL;
    // if (!check_dims_equal(image, mask)) return NULL;

    /*  construct the output array, like the input image array */
    out_array = PyArray_NewLikeArray(image, NPY_ANYORDER, NULL, 0);
    if (out_array == NULL)
        return NULL;
    PyArray_FILLWBYTE(out_array, 0);

    convolve(image, mask, kernel, out_array);
    // npy_intp size = PyArray_DIM(image, 0);
    // for (int i=0; i<size; ++i)
    // {
    //     double old_num = get_double(image, i);
    //     set_double(out_array, i, old_num*10);
    //     // double* new_num = (double *) PyArray_GETPTR1(out_array, i);
    //     // *new_num = old_num * 10;
    // }

    // /*  create the iterators */
    // in_iter = NpyIter_New(in_array, NPY_ITER_READONLY, NPY_KEEPORDER,
    //                          NPY_NO_CASTING, NULL);
    // if (in_iter == NULL)
    //     goto fail;
    //
    // out_iter = NpyIter_New((PyArrayObject *)out_array, NPY_ITER_READWRITE,
    //                       NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    // if (out_iter == NULL) {
    //     NpyIter_Deallocate(in_iter);
    //     goto fail;
    // }
    //
    // in_iternext = NpyIter_GetIterNext(in_iter, NULL);
    // out_iternext = NpyIter_GetIterNext(out_iter, NULL);
    // if (in_iternext == NULL || out_iternext == NULL) {
    //     NpyIter_Deallocate(in_iter);
    //     NpyIter_Deallocate(out_iter);
    //     goto fail;
    // }
    // double ** in_dataptr = (double **) NpyIter_GetDataPtrArray(in_iter);
    // double ** out_dataptr = (double **) NpyIter_GetDataPtrArray(out_iter);
    //
    // /*  iterate over the arrays */
    // do {
    //     **out_dataptr = cos(**in_dataptr);
    // } while(in_iternext(in_iter) && out_iternext(out_iter));
    //
    /*  clean up and return the result */
    // NpyIter_Deallocate(in_iter);
    // NpyIter_Deallocate(out_iter);
    Py_INCREF(out_array);
    return out_array;

    /*  in case bad things happen */
    fail:
        Py_XDECREF(out_array);
        return NULL;
}

/*  define functions in module */
static PyMethodDef module_methods[] =
{
     {"deformable_covolution", deformable_covolution, METH_VARARGS,
         "evaluate the cosine on a numpy array"},
     {NULL, NULL, 0, NULL}
};

/* module initialization */
PyMODINIT_FUNC initconvolve_tools(void)
{
     (void) Py_InitModule("convolve_tools", module_methods);
     /* IMPORTANT: this must be called */
     import_array();
}

/*
Deformable convolution Python module.

The original code for the convolution is largely based on that of Song Ho An.
http://www.songho.ca/dsp/convolution/convolution.html
Accessed: 14/03/2015

*/

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

#define QUOTE(s) # s
#define get_double(a, i, j) *((double *) PyArray_GETPTR2(a, i, j))
#define set_double(a, i, j, value) *((double *) PyArray_GETPTR2(a, i, j)) = value;

#define get_int(a, i, j) *((int *) PyArray_GETPTR2(a, i, j))
#define set_int(a, i, j, value) *((int *) PyArray_GETPTR2(a, i, j)) = value;


void convolve(PyArrayObject* image, PyArrayObject* mask, PyArrayObject* kernel, PyObject* out)
{
    int rows = (int) PyArray_DIM(image, 0);
    int cols = (int) PyArray_DIM(image, 1);
    npy_intp kernelSize = PyArray_DIM(kernel, 0);

    int kRows = (int) kernelSize;
    int kCols = (int) kernelSize;

    // find center position of kernel (half of kernel size)
    int kCenterX = kCols / 2;
    int kCenterY = kRows / 2;

    for(int i=0; i < rows; ++i)              // rows
    {
        for(int j=0; j < cols; ++j)          // columns
        {

            int nonzero_count = 0;
            double norm_sum = 0;

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
                        double value = get_double(kernel, mm, nn);
                        double mask_value = get_double(mask, ii, jj);

                        if (mask_value > 0)
                        {
                            nonzero_count++;
                            norm_sum += value;
                        }
                    }
                }
            }

            //ignore areas which are empty
            if (nonzero_count == 0)
            {
                continue;
            }
            //Kernel has been deformed and is therefore no longer seperable
            else
            {
                double value = get_double(out, i, j);
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
                            double kernel_value = get_double(kernel, mm, nn);
                            double mask_value = get_double(mask, ii, jj);
                            //normalise value by local neighbourhood
                            kernel_value -= norm_sum / nonzero_count;
                            ///weight by the mask value
                            kernel_value *= mask_value;
                            //convolve with image pixel
                            value += get_double(image,ii,jj) * kernel_value;
                        }
                    }
                }
                set_double(out, i, j, value);
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

    /*  parse the image, mask and kernel */
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &image, &PyArray_Type, &mask, &PyArray_Type, &kernel))
        return NULL;

    /*  construct the output array, like the input image array */
    out_array = PyArray_NewLikeArray(image, NPY_ANYORDER, NULL, 0);
    if (out_array == NULL)
        return NULL;
    PyArray_FILLWBYTE(out_array, 0);

    convolve(image, mask, kernel, out_array);

    Py_INCREF(out_array);
    return out_array;
}

/*  define functions in module */
static PyMethodDef module_methods[] =
{
     {"deformable_covolution", deformable_covolution, METH_VARARGS,
         "Convolve an image with a kernel and a mask. Areas outside the mask will be"
         "ignored by the convolution."},
     {NULL, NULL, 0, NULL}
};

/* module initialization */
PyMODINIT_FUNC initconvolve_tools(void)
{
     (void) Py_InitModule("convolve_tools", module_methods);
     /* IMPORTANT: this must be called */
     import_array();
}

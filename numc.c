#include "numc.h"
#include <structmember.h>

PyTypeObject Matrix61cType;

/* Helper functions for initalization of matrices and vectors */

/*
 * Return a tuple given rows and cols
 */
PyObject *get_shape(int rows, int cols) {
  if (rows == 1 || cols == 1) {
    return PyTuple_Pack(1, PyLong_FromLong(rows * cols));
  } else {
    return PyTuple_Pack(2, PyLong_FromLong(rows), PyLong_FromLong(cols));
  }
}
/*
 * Matrix(rows, cols, low, high). Fill a matrix random double values
 */
int init_rand(PyObject *self, int rows, int cols, unsigned int seed, double low,
              double high) {
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    rand_matrix(new_mat, seed, low, high);
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * Matrix(rows, cols, val). Fill a matrix of dimension rows * cols with val
 */
int init_fill(PyObject *self, int rows, int cols, double val) {
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed)
        return alloc_failed;
    else {
        fill_matrix(new_mat, val);
        ((Matrix61c *)self)->mat = new_mat;
        ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    }
    return 0;
}

/*
 * Matrix(rows, cols, 1d_list). Fill a matrix with dimension rows * cols with 1d_list values
 */
int init_1d(PyObject *self, int rows, int cols, PyObject *lst) {
    if (rows * cols != PyList_Size(lst)) {
        PyErr_SetString(PyExc_ValueError, "Incorrect number of elements in list");
        return -1;
    }
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    int count = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            set(new_mat, i, j, PyFloat_AsDouble(PyList_GetItem(lst, count)));
            count++;
        }
    }
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * Matrix(2d_list). Fill a matrix with dimension len(2d_list) * len(2d_list[0])
 */
int init_2d(PyObject *self, PyObject *lst) {
    int rows = PyList_Size(lst);
    if (rows == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot initialize numc.Matrix with an empty list");
        return -1;
    }
    int cols;
    if (!PyList_Check(PyList_GetItem(lst, 0))) {
        PyErr_SetString(PyExc_ValueError, "List values not valid");
        return -1;
    } else {
        cols = PyList_Size(PyList_GetItem(lst, 0));
    }
    for (int i = 0; i < rows; i++) {
        if (!PyList_Check(PyList_GetItem(lst, i)) ||
                PyList_Size(PyList_GetItem(lst, i)) != cols) {
            PyErr_SetString(PyExc_ValueError, "List values not valid");
            return -1;
        }
    }
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            set(new_mat, i, j,
                PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(lst, i), j)));
        }
    }
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * This deallocation function is called when reference count is 0
 */
void Matrix61c_dealloc(Matrix61c *self) {
    deallocate_matrix(self->mat);
    Py_TYPE(self)->tp_free(self);
}

/* For immutable types all initializations should take place in tp_new */
PyObject *Matrix61c_new(PyTypeObject *type, PyObject *args,
                        PyObject *kwds) {
    /* size of allocated memory is tp_basicsize + nitems*tp_itemsize*/
    Matrix61c *self = (Matrix61c *)type->tp_alloc(type, 0);
    return (PyObject *)self;
}

/*
 * This matrix61c type is mutable, so needs init function. Return 0 on success otherwise -1
 */
int Matrix61c_init(PyObject *self, PyObject *args, PyObject *kwds) {
    /* Generate random matrices */
    if (kwds != NULL) {
        PyObject *rand = PyDict_GetItemString(kwds, "rand");
        if (!rand) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
        if (!PyBool_Check(rand)) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
        if (rand != Py_True) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }

        PyObject *low = PyDict_GetItemString(kwds, "low");
        PyObject *high = PyDict_GetItemString(kwds, "high");
        PyObject *seed = PyDict_GetItemString(kwds, "seed");
        double double_low = 0;
        double double_high = 1;
        unsigned int unsigned_seed = 0;

        if (low) {
            if (PyFloat_Check(low)) {
                double_low = PyFloat_AsDouble(low);
            } else if (PyLong_Check(low)) {
                double_low = PyLong_AsLong(low);
            }
        }

        if (high) {
            if (PyFloat_Check(high)) {
                double_high = PyFloat_AsDouble(high);
            } else if (PyLong_Check(high)) {
                double_high = PyLong_AsLong(high);
            }
        }

        if (double_low >= double_high) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }

        // Set seed if argument exists
        if (seed) {
            if (PyLong_Check(seed)) {
                unsigned_seed = PyLong_AsUnsignedLong(seed);
            }
        }

        PyObject *rows = NULL;
        PyObject *cols = NULL;
        if (PyArg_UnpackTuple(args, "args", 2, 2, &rows, &cols)) {
            if (rows && cols && PyLong_Check(rows) && PyLong_Check(cols)) {
                return init_rand(self, PyLong_AsLong(rows), PyLong_AsLong(cols), unsigned_seed, double_low,
                                 double_high);
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
    }
    PyObject *arg1 = NULL;
    PyObject *arg2 = NULL;
    PyObject *arg3 = NULL;
    if (PyArg_UnpackTuple(args, "args", 1, 3, &arg1, &arg2, &arg3)) {
        /* arguments are (rows, cols, val) */
        if (arg1 && arg2 && arg3 && PyLong_Check(arg1) && PyLong_Check(arg2) && (PyLong_Check(arg3)
                || PyFloat_Check(arg3))) {
            if (PyLong_Check(arg3)) {
                return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), PyLong_AsLong(arg3));
            } else
                return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), PyFloat_AsDouble(arg3));
        } else if (arg1 && arg2 && arg3 && PyLong_Check(arg1) && PyLong_Check(arg2) && PyList_Check(arg3)) {
            /* Matrix(rows, cols, 1D list) */
            return init_1d(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), arg3);
        } else if (arg1 && PyList_Check(arg1) && arg2 == NULL && arg3 == NULL) {
            /* Matrix(rows, cols, 1D list) */
            return init_2d(self, arg1);
        } else if (arg1 && arg2 && PyLong_Check(arg1) && PyLong_Check(arg2) && arg3 == NULL) {
            /* Matrix(rows, cols, 1D list) */
            return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), 0);
        } else {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return -1;
    }
}

/*
 * List of lists representations for matrices
 */
PyObject *Matrix61c_to_list(Matrix61c *self) {
    int rows = self->mat->rows;
    int cols = self->mat->cols;
    PyObject *py_lst = NULL;
    if (self->mat->is_1d) {  // If 1D matrix, print as a single list
        py_lst = PyList_New(rows * cols);
        int count = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                PyList_SetItem(py_lst, count, PyFloat_FromDouble(get(self->mat, i, j)));
                count++;
            }
        }
    } else {  // if 2D, print as nested list
        py_lst = PyList_New(rows);
        for (int i = 0; i < rows; i++) {
            PyList_SetItem(py_lst, i, PyList_New(cols));
            PyObject *curr_row = PyList_GetItem(py_lst, i);
            for (int j = 0; j < cols; j++) {
                PyList_SetItem(curr_row, j, PyFloat_FromDouble(get(self->mat, i, j)));
            }
        }
    }
    return py_lst;
}

PyObject *Matrix61c_class_to_list(Matrix61c *self, PyObject *args) {
    PyObject *mat = NULL;
    if (PyArg_UnpackTuple(args, "args", 1, 1, &mat)) {
        if (!PyObject_TypeCheck(mat, &Matrix61cType)) {
            PyErr_SetString(PyExc_TypeError, "Argument must of type numc.Matrix!");
            return NULL;
        }
        Matrix61c* mat61c = (Matrix61c*)mat;
        return Matrix61c_to_list(mat61c);
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return NULL;
    }
}

/*
 * Add class methods
 */
PyMethodDef Matrix61c_class_methods[] = {
    {"to_list", (PyCFunction)Matrix61c_class_to_list, METH_VARARGS, "Returns a list representation of numc.Matrix"},
    {NULL, NULL, 0, NULL}
};

/*
 * Matrix61c string representation. For printing purposes.
 */
PyObject *Matrix61c_repr(PyObject *self) {
    PyObject *py_lst = Matrix61c_to_list((Matrix61c *)self);
    return PyObject_Repr(py_lst);
}

/* NUMBER METHODS */

/*
 * Add the second numc.Matrix (Matrix61c) object to the first one. The first operand is
 * self, and the second operand can be obtained by casting `args`.
 */
PyObject *Matrix61c_add(Matrix61c* self, PyObject* args) {
    /* TODO: YOUR CODE HERE */
    
    if (!PyObject_TypeCheck(args, &Matrix61cType)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be of type numc.Matrix!");
        return NULL;
    }
    Matrix61c* mat61c = (Matrix61c*)args;
    
    //check if dimesions are same
    if ((self->mat->rows!=mat61c->mat->rows) || (self->mat->cols!=mat61c->mat->cols)) {
        PyErr_SetString(PyExc_ValueError, "matrices must have same dimensions");
        return NULL;
    }
    
    Matrix61c *rv = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
    matrix *new_mat;
    rv->mat = new_mat;
    /* Set the shape of this numc.Matrix */
    rv->shape = get_shape(new_mat->rows, new_mat->cols);
    int alloc_failed = allocate_matrix(&new_mat, self->mat->rows, self->mat->cols);
    if (alloc_failed){
        Matrix61c_dealloc(rv);
        PyErr_SetString(PyExc_RuntimeError, "add_matrix alloc failed");
    }
    int add_failed = add_matrix(new_mat, self->mat, mat61c->mat);
    if (add_failed) {
        return NULL;
    }
    
    return (PyObject*) rv;
}

/*
 * Substract the second numc.Matrix (Matrix61c) object from the first one. The first operand is
 * self, and the second operand can be obtained by casting `args`.
 */
PyObject *Matrix61c_sub(Matrix61c* self, PyObject* args) {
    /* TODO: YOUR CODE HERE */
    if (!PyObject_TypeCheck(args, &Matrix61cType)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be of type numc.Matrix!");
        return NULL;
    }
    Matrix61c* mat61c = (Matrix61c*)args;
    
    //check if dimesions are same
    if ((self->mat->rows!=mat61c->mat->rows) || (self->mat->cols!=mat61c->mat->cols)) {
        PyErr_SetString(PyExc_ValueError, "matrices must have same dimensions");
        return NULL;
    }
    
    Matrix61c *rv = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
    matrix *new_mat;
    rv->mat = new_mat;
    /* Set the shape of this numc.Matrix */
    rv->shape = get_shape(new_mat->rows, new_mat->cols);
    int alloc_failed = allocate_matrix(&new_mat, self->mat->rows, self->mat->cols);
    if (alloc_failed){
        Matrix61c_dealloc(rv);
        PyErr_SetString(PyExc_RuntimeError, "sub_matrix alloc failed");
    }
    int sub_failed = sub_matrix(new_mat, self->mat, mat61c->mat);
    if (sub_failed) {
        return NULL;
    }
    
    return (PyObject*) rv;
}

/*
 * NOT element-wise multiplication. The first operand is self, and the second operand
 * can be obtained by casting `args`.
 */
PyObject *Matrix61c_multiply(Matrix61c* self, PyObject *args) {
    /* TODO: YOUR CODE HERE */
    
    //obtain second matrix
    if (!PyObject_TypeCheck(args, &Matrix61cType)) {
        PyErr_SetString(PyExc_TypeError, "Argument must of type numc.Matrix!");
        return NULL;
    }
    Matrix61c* mat61c = (Matrix61c*)args;
    
    //check if dimesions are same
    if (self->mat->cols!=mat61c->mat->rows) {
        PyErr_SetString(PyExc_ValueError, "mat1's col must be same as mat2's rows");
        return NULL;
    }
    
    Matrix61c *rv = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
    matrix *new_mat;
    rv->mat = new_mat;
    /* Set the shape of this numc.Matrix */
    rv->shape = get_shape(new_mat->rows, new_mat->cols);
    int alloc_failed = allocate_matrix(&new_mat, self->mat->rows, self->mat->cols);
    if (alloc_failed){
        Matrix61c_dealloc(rv);
        PyErr_SetString(PyExc_RuntimeError, "mul_matrix alloc failed");
    }
    int mul_failed = mul_matrix(new_mat, self->mat, mat61c->mat);
    if (mul_failed) {
        return NULL;
    }
    return (PyObject*) rv;
    
}

/*
 * Negates the given numc.Matrix.
 */
PyObject *Matrix61c_neg(Matrix61c* self) {
    /* TODO: YOUR CODE HERE */
    Matrix61c *rv = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
    matrix *new_mat;
    rv->mat = new_mat;
    /* Set the shape of this numc.Matrix */
    rv->shape = get_shape(new_mat->rows, new_mat->cols);
    int alloc_failed = allocate_matrix(&new_mat, self->mat->rows, self->mat->cols);
    if (alloc_failed){
        Matrix61c_dealloc(rv);
        PyErr_SetString(PyExc_RuntimeError, "neg_matrix alloc failed");
    }
    int neg_failed = neg_matrix(new_mat, self->mat);
    if (neg_failed) {
        return NULL;
    }
    
    return (PyObject*) rv;
}

/*
 * Take the element-wise absolute value of this numc.Matrix.
 */
PyObject *Matrix61c_abs(Matrix61c *self) {
    /* TODO: YOUR CODE HERE */
    Matrix61c *rv = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
    matrix *new_mat;
    rv->mat = new_mat;
    /* Set the shape of this numc.Matrix */
    rv->shape = get_shape(new_mat->rows, new_mat->cols);
    int alloc_failed = allocate_matrix(&new_mat, self->mat->rows, self->mat->cols);
    if (alloc_failed){
        Matrix61c_dealloc(rv);
        PyErr_SetString(PyExc_RuntimeError, "abs_matrix alloc failed");
        
    }
    int abs_failed = abs_matrix(new_mat, self->mat);
    if (abs_failed) {
        // PyErr_SetString(PyExc_RuntimeError, "Abs_matrix failed.");
        return NULL;
    }
    
    return (PyObject*) rv;
}

/*
 * Raise numc.Matrix (Matrix61c) to the `pow`th power. You can ignore the argument `optional`.
 */
PyObject *Matrix61c_pow(Matrix61c *self, PyObject *pow, PyObject *optional) {
    /* TODO: YOUR CODE HERE */
    
    if (!PyLong_Check(pow)){
        PyErr_SetString(PyExc_TypeError, "Argument pow must be of type int");
        return NULL;
    }
    int power = PyLong_AsLong(pow);
    if (self->mat->cols!=self->mat->rows) {
        PyErr_SetString(PyExc_ValueError, "matrix must be square");
        return NULL;
    }
    if (power<0) {
        PyErr_SetString(PyExc_ValueError, "pow must be greater than 0");
        return NULL;
    }
    
    
    Matrix61c *rv = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
    matrix *new_mat;
    rv->mat = new_mat;
    //Set the shape of this numc.Matrix
    rv->shape = get_shape(new_mat->rows, new_mat->cols);
    int alloc_failed = allocate_matrix(&new_mat, self->mat->rows, self->mat->cols);
    if (alloc_failed){
        Matrix61c_dealloc(rv);
        PyErr_SetString(PyExc_RuntimeError, "pow_matrix alloc failed");
    }
    int pow_failed = pow_matrix(new_mat, self->mat, power);
    if (pow_failed) {
        PyErr_SetString(PyExc_RuntimeError, "pow_matrix failed.");
        return NULL;
    }
    
    return (PyObject*) rv;
}

/*
 * Create a PyNumberMethods struct for overloading operators with all the number methods you have
 * define. You might find this link helpful: https://docs.python.org/3.6/c-api/typeobj.html
 */
PyNumberMethods Matrix61c_as_number = {
    /* TODO: YOUR CODE HERE */
    .nb_add = (binaryfunc) Matrix61c_add,
    .nb_subtract = (binaryfunc) Matrix61c_sub,
    .nb_matrix_multiply = (binaryfunc) Matrix61c_multiply,
    .nb_power = (ternaryfunc) Matrix61c_pow,
    .nb_negative = (unaryfunc) Matrix61c_neg,
    .nb_absolute = (unaryfunc) Matrix61c_abs
};


/* INSTANCE METHODS */

/*
 * Given a numc.Matrix self, parse `args` to (int) row, (int) col, and (double/int) val.
 * Return None in Python (this is different from returning null).
 */
PyObject *Matrix61c_set_value(Matrix61c *self, PyObject* args) {
    /* TODO: YOUR CODE HERE */
    int row;
    int col;
    double val;
    if (PyArg_ParseTuple(args, "iid", &row, &col, &val)) {
        if (row >= self->mat->rows || col >= self->mat->cols) {
            PyErr_SetString(PyExc_IndexError, "i and j must be within matrix size");
            return NULL;
        }
        // int set_failed = 
        set(self->mat, row, col, val);    
        // Py_RETURN_NONE;
        return Py_None;
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return NULL;
    }
}

/*
 * Given a numc.Matrix `self`, parse `args` to (int) row and (int) col.
 * Return the value at the `row`th row and `col`th column, which is a Python
 * float/int.
 */
PyObject *Matrix61c_get_value(Matrix61c *self, PyObject* args) {
    /* TODO: YOUR CODE HERE */
    int row;
    int col;
    if (PyArg_ParseTuple(args, "ii", &row, &col)) {
        if (row >= self->mat->rows || col >= self->mat->cols) {
            PyErr_SetString(PyExc_IndexError, "i and j must be within matrix size");
            return NULL;
        }
        double val = get(self->mat, row, col);
        return PyFloat_FromDouble(val);
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return NULL;
    }
    // return PyFloat_FromDouble(0.0);
}

/*
 * Create an array of PyMethodDef structs to hold the instance methods.
 * Name the python function corresponding to Matrix61c_get_value as "get" and Matrix61c_set_value
 * as "set"
 * You might find this link helpful: https://docs.python.org/3.6/c-api/structures.html
 */
PyMethodDef Matrix61c_methods[] = {
    /* TODO: YOUR CODE HERE */
    {"set", (PyCFunction) Matrix61c_set_value, METH_VARARGS, "set method"}, 
    {"get", (PyCFunction) &Matrix61c_get_value, METH_VARARGS, "get method"}, 
    {NULL, NULL, 0, NULL}
};

/* INDEXING */

/*
 * Given a numc.Matrix `self`, index into it with `key`. Return the indexed result.
 */
PyObject *Matrix61c_subscript(Matrix61c* self, PyObject* key) {
    /* TODO: YOUR CODE HERE */
    //1D Matrix
    if (self->mat->is_1d) {
        if (PyLong_Check(key)) {
            if (PyLong_asLong(key) >= self->mat->is_1d) {
                PyErr_SetString(PyExc_IndexError, "Integer key out of bounds for 1D matrix");
                return NULL;
            }
            //if else case for whether rows==1 or cols==1
            if (self->mat->rows == 1) {
                return self->mat->data[0][PyLong_asLong(key)];
            } else {
                return self->mat->data[PyLong_asLong(key)][0];
            }
        } else if (PySlice_Check(key)) {
            Py_ssize_t start;
            Py_ssize_t stop;
            Py_ssize_t step;
            Py_ssize_t slicelength;
            //error checking
            if (PySlice_GetIndicesEx(key, self->mat->is_1d, &start, &stop, &step, &slicelength) < 0) {
                return NULL;
            }
            if (step != 1) {
                PyErr_SetString(PyExc_ValueError, "Slice step size can only be 1");
                return NULL;
            }
            if (slicelength < 1) {
                PyErr_SetString(PyExc_ValueError, "Slice length can't be < 1");
                return NULL;            
            }
            if (slicelength == 1) {
                if (self->mat->rows == 1) {
                    return self->mat->data[0][start];
                } else {
                    return self->mat->data[start][0];
                }
            }
            //needs a new matrix
            matrix *mat1 = NULL;
            int num = stop - start;
            int alloc_failed;
            if (self->mat->rows == 1) {
                alloc_failed = allocate_matrix_ref(mat1, self->mat, 0, start, 1, num);
                if (alloc_failed != 0) {
                    return NULL;
                }
                //return the matrix
                Matrix61c *rv = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
                rv->mat = mat1;
                /* Set the shape of this numc.Matrix */
                rv->shape = get_shape(mat1->rows, mat1->cols);
                return (PyObject*) rv;
            } else {
                alloc_failed = allocate_matrix_ref(mat1, self->mat, start, 0, num, 1);
                if (alloc_failed != 0) {
                    return NULL;
                }
                //return the matrix
                Matrix61c *rv = (Matrix61c *) Matrix61c_new(&Matrix61cType, NULL, NULL);
                rv->mat = mat1;
                /* Set the shape of this numc.Matrix */
                rv->shape = get_shape(mat1->rows, mat1->cols);
                return (PyObject*) rv;            
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Invalid key for 1D matrix");
            return NULL;
        }   
    } else {
        if (PyLong_Check(key)) {
            if ()
        } else if (PySlice_Check(key)) {

        } else if (PyTuple_Check(key)) {
            
        } else {
            PyErr_SetString(PyExc_TypeError, "Invalid key for 2D matrix");
            return NULL;
        }
    }
     PySlice_GetIndicesEx

     // PyObject *get_shape(int rows, int cols)
}

/*
 * Given a numc.Matrix `self`, index into it with `key`, and set the indexed result to `v`.
 */
int Matrix61c_set_subscript(Matrix61c* self, PyObject *key, PyObject *v) {
    /* TODO: YOUR CODE HERE */
    
    //is key a number?? What the hell is key KEY?
    if (!PyLong_Check(key)&&!PySlice_Check(key)&&!PyTuple_Check(v)) {
        PyErr_SetString(PyExc_TypeError, "v must be int, slice, or tuple");
        return -1;
    }
    //retrieve information of slice for PySlice_GetIndicesEx()
    
    
    if (self->mat->is_1d) {
        int mat_len = (self->mat->rows > self->mat->cols) ? self->mat->rows : self->mat->cols;
        Py_ssize_t *start;
        Py_ssize_t *stop;
        Py_ssize_t *step;
        Py_ssize_t *slicelength;
        //not sure whether length should be exclusively mat->rows or whichever one is greater than one
        PySlice_GetIndicesEx(key, mat_len, start, stop, step, slicelength);
        /*ValueError if
         - Resulting slice is 1D, but v has the wrong length, or if any element of v is not a float or int.*/
        if (){
            PyErr_SetString(PyExc_ValueError, "v is wrong length");
            return -1;
        }
        if (){
            PyErr_SetString(PyExc_ValueError, "values in v must be float/int");
            return -1;
        }
        if (){
            PyErr_SetString(PyExc_TypeError, "Resulting slice is 1 by 1, but v is not a float or int.");
            return -1;
        }
        if (){
            PyErr_SetString(PyExc_TypeError, "Resulting slice is not 1 by 1, but v is not a list.");
            return -1;
        }
        
    } else {
        Py_ssize_t *start;
        Py_ssize_t *stop;
        Py_ssize_t *step;
        Py_ssize_t *slicelength;
        PySlice_GetIndicesEx(key,self->mat->rows, start, stop, step, slicelength);
        /*ValueError if
         - Resulting slice is 2D, but v has the wrong length, or if any element of v has the wrong length, or if any element of an element of v is not a float or int.*/
        if (){
            PyErr_SetString(PyExc_ValueError, "v is wrong length");
            return -1;
        }
        if (){
            PyErr_SetString(PyExc_ValueError, "values in v must be float/int");
            return -1;
        }
        if (){
            PyErr_SetString(PyExc_TypeError, "Resulting slice is 1 by 1, but v is not a float or int.");
            return -1;
        }
        if (){
            PyErr_SetString(PyExc_TypeError, "Resulting slice is not 1 by 1, but v is not a list.");
            return -1;
        }
    }
    
    
    
    /*
     TypeError if
     - Resulting slice is 1 by 1, but v is not a float or int.
     - Resulting slice is not 1 by 1, but v is not a list.
     ValueError if
     - Resulting slice is 1D, but v has the wrong length, or if any element of v is not a float or int.
     - Resulting slice is 2D, but v has the wrong length, or if any element of v has the wrong length, or if any element of an element of v is not a float or int.
     */
    //if nothing wrong, then return 0
    return 0;
}

PyMappingMethods Matrix61c_mapping = {
    NULL,
    (binaryfunc) Matrix61c_subscript,
    (objobjargproc) Matrix61c_set_subscript,
};

/* INSTANCE ATTRIBUTES*/
PyMemberDef Matrix61c_members[] = {
    {
        "shape", T_OBJECT_EX, offsetof(Matrix61c, shape), 0,
        "(rows, cols)"
    },
    {NULL}  /* Sentinel */
};

PyTypeObject Matrix61cType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numc.Matrix",
    .tp_basicsize = sizeof(Matrix61c),
    .tp_dealloc = (destructor)Matrix61c_dealloc,
    .tp_repr = (reprfunc)Matrix61c_repr,
    .tp_as_number = &Matrix61c_as_number,
    .tp_flags = Py_TPFLAGS_DEFAULT |
    Py_TPFLAGS_BASETYPE,
    .tp_doc = "numc.Matrix objects",
    .tp_methods = Matrix61c_methods,
    .tp_members = Matrix61c_members,
    .tp_as_mapping = &Matrix61c_mapping,
    .tp_init = (initproc)Matrix61c_init,
    .tp_new = Matrix61c_new
};


struct PyModuleDef numcmodule = {
    PyModuleDef_HEAD_INIT,
    "numc",
    "Numc matrix operations",
    -1,
    Matrix61c_class_methods
};

/* Initialize the numc module */
PyMODINIT_FUNC PyInit_numc(void) {
    PyObject* m;

    if (PyType_Ready(&Matrix61cType) < 0)
        return NULL;

    m = PyModule_Create(&numcmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&Matrix61cType);
    PyModule_AddObject(m, "Matrix", (PyObject *)&Matrix61cType);
    printf("CS61C Fall 2020 Project 4: numc imported!\n");
    fflush(stdout);
    return m;
}

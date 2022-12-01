#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/*
 * Generates a random double between `low` and `high`.
 */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/*
 * Generates a random matrix with `seed`.
 */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocate space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. Remember to set all fieds of the matrix struct.
 * `parent` should be set to NULL to indicate that this matrix is not a slice.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. If you don't set python error messages here upon
 * failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    if (!(rows>0) || !(cols>0)){
        PyErr_SetString(PyExc_ValueError, "index out of bounds (needs to at least have 1 col and 1 row)");
        return -1;
    }
    
    matrix* filler = (matrix*) calloc(1, sizeof(matrix));
    if (filler==NULL){
        PyErr_SetString(PyExc_RuntimeError, "alloc failed");
        deallocate_matrix(filler);
        return -1;
    }
    
    *mat = filler;
    double** data = (double**) _mm_malloc(rows, sizeof(double*));
    (filler)->data = data;
    if (data == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "alloc failed");
        deallocate_matrix(*mat);
        return -1;
    }
    int i;
    for (i=0; i<rows; i++){
        double* temp = (double*) _mm_malloc(cols, sizeof(double));
        if (temp == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "alloc failed");
            deallocate_matrix(*mat);
            return -1;
        }
        data[i] = temp;
    }

    
    /* REF_CNT IS SET TO ZERO*/
    (filler)->rows = rows;
    (filler)->cols = cols;
    (filler)->is_1d = (rows==1 || cols==1)? rows*cols : 0;
    (filler)->ref_cnt = 1;
    (filler)->parent = NULL;
    fill_matrix(filler, 0);

    return 0;
}

/*
 * Allocate space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * This is equivalent to setting the new matrix to be
 * from[row_offset:row_offset + rows, col_offset:col_offset + cols]
 * If you don't set python error messages here upon failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int row_offset, int col_offset,
                        int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    if((from->rows < rows+row_offset) || (from->cols < cols+col_offset)) {
        PyErr_SetString(PyExc_ValueError, "index out of bounds (splice cannot be bigger than reference matrix size)");
        return -1;
    }
    
    if (!(rows>0) || !(cols>0)){
        PyErr_SetString(PyExc_ValueError, "index out of bounds (needs to at least have 1 col and 1 row)");
        return -1;
    }
    
    matrix* filler = _mm_malloc(1, sizeof(matrix));
    if (filler == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "alloc failed");
        deallocate_matrix(filler);
        return -1;
    }
    filler->rows = rows;
    filler->cols = cols;
    filler->is_1d = (rows==1 || cols==1)? rows*cols : 0;
    filler->ref_cnt = 1;
    
    double** data = (double**) _mm_malloc(rows, sizeof(double*));
    *mat = filler;
    (filler)->data = data;
    if (data == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "alloc failed");
        deallocate_matrix(*mat);
        return -1;
    }
    double** ref = from->data;
    for (int r = 0; r <rows; r++) {
        data[r] = (ref[r+row_offset]+(col_offset*sizeof(double))); //do we need to col_offset*4 for size of double?
    }
    
    from->ref_cnt++;
    filler->parent = from;
    

    return 0;
}

/*
 * This function will be called automatically by Python when a numc matrix loses all of its
 * reference pointers.
 * You need to make sure that you only free `mat->data` if no other existing matrices are also
 * referring this data array.
 * See the spec for more information.
 */
void deallocate_matrix(matrix *mat) {
    /* TODO: YOUR CODE HERE */
    
    if (mat!=NULL && mat->ref_cnt == 1) {
        //if has no parent and has no children
        if (mat->parent==NULL){
            double** data = mat->data;
            if(data != NULL) {
                for(int r = 0; r<mat->rows; r++){
                    if (data[r]) {
                        _mm_free(data[r]);
                    }
                }
                _mm_free(data);
            }
            _mm_free(mat);
        } else {
            //if child
            if (mat->parent!=NULL) {
                mat->parent->ref_cnt-=1;
                _mm_free(mat->data);
                _mm_free(mat);
            }
            //cannot delete if is parent
        }
        
    }
    
}

/*
 * Return the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    /* TODO: YOUR CODE HERE */
    double** matrix = mat->data;
    double* rowOfData = matrix[row];
    return rowOfData[col];
}

/*
 * Set the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    /* TODO: YOUR CODE HERE */
    mat->data[row][col] = val;
}

/*
 * Set all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    /* TODO: YOUR CODE HERE */
    __m256d temp = _mm256_set1_pd (val);
    for (int r = 0; r < mat->rows; r++) {
        for (int c = 0; c < mat->cols/16*16; c+=16){
            _mm256_storeu_pd((*(mat->data+r)+c), temp);
            _mm256_storeu_pd((*(mat->data+r)+c+4), temp);
            _mm256_storeu_pd((*(mat->data+r)+c+8), temp);
            _mm256_storeu_pd((*(mat->data+r)+c+12), temp);
        }
        
        for (int c = mat->cols/16*16; c < mat->cols/4*4; c+=4){
            _mm256_storeu_pd((*(mat->data+r)+c), temp);
        }
        for (int c = mat->cols/4*4; c<mat->cols; c++){
            set(mat, r, c, val);
        }
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    
    __m256d res = _mm256_set1_pd(0);
    double** data1 = mat1->data;
    double** data2 = mat2->data;
    for (int r = 0; r < mat1->rows; r++) {
        for (int c = 0; c < mat1->cols/16*16; c+=16){
            __m256d a1 = _mm256_loadu_pd(*(data1+r)+c);
            __m256d a2  = _mm256_loadu_pd(*(data2+r)+c);
            res = _mm256_add_pd(a1, a2);
            _mm256_storeu_pd((*(result->data+r)+c), res);
            
            __m256d b1 = _mm256_loadu_pd(*(data1+r)+c+4);
            __m256d b2  = _mm256_loadu_pd(*(data2+r)+c+4);
            res = _mm256_add_pd(b1, b2);
            _mm256_storeu_pd((*(result->data+r)+c+4), res);
            
            __m256d c1 = _mm256_loadu_pd(*(data1+r)+c+8);
            __m256d c2  = _mm256_loadu_pd(*(data2+r)+c+8);
            res = _mm256_add_pd(c1, c2);
            _mm256_storeu_pd((*(result->data+r)+c+8), res);
            
            __m256d d1 = _mm256_loadu_pd(*(data1+r)+c+12);
            __m256d d2  = _mm256_loadu_pd(*(data2+r)+c+12);
            res = _mm256_add_pd(d1, d2);
            _mm256_storeu_pd((*(result->data+r)+c+12), res);
        }
        
        for (int c = mat1->cols/16*16; c < mat1->cols/4*4; c+=4) {
            __m256d d1 = _mm256_loadu_pd(*(data1+r)+c);
            __m256d d2  = _mm256_loadu_pd(*(data2+r)+c);
            res = _mm256_add_pd(d1, d2);
            _mm256_storeu_pd((*(result->data+r)+c), res);
        }
        
        for (int c = mat1->cols/4*4; c<mat1->cols; c++){
            result->data[r][c] = data1[r][c] + data2[r][c];
        }
    }
    
    return 0;
    

}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    /*what cases make it fail?
        1. if row and col values not same
     2.
     */
    __m256d res = _mm256_set1_pd(0);
    double** data1 = mat1->data;
    double** data2 = mat2->data;
    for (int r = 0; r < mat1->rows; r++) {
        for (int c = 0; c < mat1->cols/16*16; c+=16){
            __m256d a1 = _mm256_loadu_pd(*(data1+r)+c);
            __m256d a2  = _mm256_loadu_pd(*(data2+r)+c);
            res = _mm256_sub_pd(a1, a2);
            _mm256_storeu_pd((*(result->data+r)+c), res);
            
            __m256d b1 = _mm256_loadu_pd(*(data1+r)+c+4);
            __m256d b2  = _mm256_loadu_pd(*(data2+r)+c+4);
            res = _mm256_sub_pd(b1, b2);
            _mm256_storeu_pd((*(result->data+r)+c+4), res);
            
            __m256d c1 = _mm256_loadu_pd(*(data1+r)+c+8);
            __m256d c2  = _mm256_loadu_pd(*(data2+r)+c+8);
            res = _mm256_sub_pd(c1, c2);
            _mm256_storeu_pd((*(result->data+r)+c+8), res);
            
            __m256d d1 = _mm256_loadu_pd(*(data1+r)+c+12);
            __m256d d2  = _mm256_loadu_pd(*(data2+r)+c+12);
            res = _mm256_sub_pd(d1, d2);
            _mm256_storeu_pd((*(result->data+r)+c+12), res);
        }
        
        for (int c = mat1->cols/16*16; c < mat1->cols/4*4; c+=4) {
            __m256d d1 = _mm256_loadu_pd(*(data1+r)+c);
            __m256d d2  = _mm256_loadu_pd(*(data2+r)+c);
            res = _mm256_sub_pd(d1, d2);
            _mm256_storeu_pd ((*(result->data+r)+c), res);
        }
        
        for (int c = mat1->cols/4*4; c<mat1->cols; c++){
            result->data[r][c] = data1[r][c] - data2[r][c];
        }
    }
    
    
    return 0;
    
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    //i'm unsure how theyre matmuling 2 matrices together
    if (mat1->cols != mat2->rows) {
    	return 1;
    }
    // if (!mat1 || !mat2) { //should we check for dim = 0 instead?
    // 	return 1;
    // }
    int n = mat1->cols;
    for (int r = 0; r < mat1->rows; r++) {
        for (int c = 0; c < mat2->cols; c++) {
            //is this int or float or double?
            float sum = 0;
            for (int i = 0; i < n; i++) {
                sum += mat1->data[r][i] * mat2->data[i][c];
            }
            result->data[r][c] = sum;
        }
    }
    // do we assume that result already has space allocated for it?
    //do we have to set cols and rows and other params of result?
    // __m256d res = _mm256_set1_pd(0);
    // double** data1 = mat1->data;
    // double** data2 = mat2->data;
    // for (int r = 0; r < mat1->rows/4*4; r+=4) {
    // 	int sum = 0;
    // 	//something about the loops is wrong
    // 	//right now we're taking the first four nums in the first row
    // 	//and multiplying them by the nums in the first col
    // 	//eg (1,1)*(1,1) (1,2)*(2,1) (1,3)*(3,1) (1,4)*(4,1) (1,1)*(5,1) (1,2)*(6,1) etc.
    // 	//we want to multiply the entire first row by the nums in the first col
    // 	//eg (1,1)*(1,1) (1,2)*(2,1) (1,3)*(3,1) (1,4)*(4,1) (1,5)*(5,1) (1,6)*(6,1) etc.

    //     for (int c = 0; c < mat1->cols/4*4; c+=4) {
    //     	int col_vals[4];
    //     	int res_arr[4];
    //         __m256d row1 = _mm256_loadu_pd(*(data1+r)+c);
    //         col_vals[0] = data2[c][r];
    //         col_vals[1] = data2[c+1][r];
    //         col_vals[2] = data2[c+2][r];
    //         col_vals[3] = data2[c+3][r];
    //         __m256d col1  = _mm256_loadu_pd(col_vals);
    //         res = _mm256_mul_pd(row1, col1);
    //         _mm256_storeu_pd(res_arr, res);
    //         for (int i = 0; i < 4; i++) {
    //         	sum += res_arr[i];
    //         }
    //         // _mm256_storeu_pd ((*(result->data+r)+c), res);
    //     }
    //     //how to sum the result and correctly store it in the result matrix?
    // }
    
    // for (int r = 0; r < mat1->rows; r++) {
    //     for (int c = mat1->cols/4*4; c<mat1->cols; c++){
    //         result->data[r][c] = data1[r][c] + data2[r][c];
    //     }
    // }
    
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    /* TODO: YOUR CODE HERE */
    if (pow<0){
        return 1;   //ERROR: SHOULD BE ZERO OR GREATER
    }

    //if pow==0, set everything to 1
    if (pow==0){
        fill_matrix(result, 1);
        return 0;
    }

    matrix *filler = NULL;
	allocate_matrix(&filler, result->rows, result->cols);

	for (int a = 0; a < result->rows; a++) {
    	for (int b = 0; b < result->cols; b++) {
    		result->data[a][b] = mat->data[a][b];
    	}
    }

    if (pow == 1) {
    	return 0;
    }

    // x^8 = ((x^2)^2)^2)
    // switch pointers instead of copying result into filler each time
    for (int i = 0; i < pow - 1; i++) {
    	for (int j = 0; j < result->rows; j++) {
    		for (int k = 0; k < result->cols; k++) {
    			filler->data[j][k] = result->data[j][k];
    		}
    	}

    	mul_matrix(result, mat, filler);
    }
    return 0;
}
    
/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    /*what cases make it fail?
        1.
     */
    __m256d zeroes = _mm256_set1_pd(0);
    __m256d res = _mm256_set1_pd(0);
    for (int r = 0; r < mat->rows; r++) {
        for (int c = 0; c < mat->cols/16*16; c+=16){
            __m256d d1 = _mm256_loadu_pd(*(mat->data+r)+c);
            res = _mm256_sub_pd(zeroes, d1);
            _mm256_storeu_pd ((*(result->data+r)+c), res);
            
            __m256d d2 = _mm256_loadu_pd(*(mat->data+r)+c+4);
            res = _mm256_sub_pd(zeroes, d2);
            _mm256_storeu_pd ((*(result->data+r)+c+4), res);
            
            __m256d d3 = _mm256_loadu_pd(*(mat->data+r)+c+8);
            res = _mm256_sub_pd(zeroes, d3);
            _mm256_storeu_pd ((*(result->data+r)+c+8), res);
            
            __m256d d4 = _mm256_loadu_pd(*(mat->data+r)+c+12);
            res = _mm256_sub_pd(zeroes, d4);
            _mm256_storeu_pd ((*(result->data+r)+c+12), res);
        }
        
        
        for (int c = 0; c < mat->cols/4*4; c+=4) {
            __m256d d1 = _mm256_loadu_pd(*(mat->data+r)+c);
            res = _mm256_sub_pd(zeroes, d1);
            _mm256_storeu_pd ((*(result->data+r)+c), res);
        }
        
        for (int c = mat->cols/4*4; c<mat->cols; c++){
            int val = mat->data[r][c];
            result->data[r][c] = (val != 0) ? 0-val : 0;
        }
    }
    
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    /*
    double** data = mat->data;
    double** res = result->data;
    for (int r = 0; r < mat->rows; r++) {
        for (int c = 0; c < mat->cols; c++) {
            int val = data[r][c];
            if (val < 0){
                val = 0-val;
            }
            res[r][c] = val;
        }
    }
    return 0;
    */
    __m256d zeroes = _mm256_set1_pd(0);
    __m256d res = _mm256_set1_pd(0);
    
    for (int r = 0; r < mat->rows; r++) {
        for (int c = 0; c < mat->cols/16*16; c+=16) {
            __m256d d1 = _mm256_loadu_pd(*(mat->data+r)+c);
            __m256d pos1 = _mm256_max_pd(zeroes, d1);
            __m256d neg1 = _mm256_max_pd(zeroes, _mm256_sub_pd(zeroes, d1));
            res = _mm256_add_pd(pos1, neg1);
            _mm256_storeu_pd ((*(result->data+r)+c), res);
            
            __m256d d2 = _mm256_loadu_pd(*(mat->data+r)+c+4);
            __m256d pos2 = _mm256_max_pd(zeroes, d2);
            __m256d neg2 = _mm256_max_pd(zeroes, _mm256_sub_pd(zeroes, d2));
            res = _mm256_add_pd(pos2, neg2);
            _mm256_storeu_pd ((*(result->data+r)+c+4), res);
            
            __m256d d3 = _mm256_loadu_pd(*(mat->data+r)+c+8);
            __m256d pos3 = _mm256_max_pd(zeroes, d3);
            __m256d neg3 = _mm256_max_pd(zeroes, _mm256_sub_pd(zeroes, d3));
            res = _mm256_add_pd(pos3, neg3);
            _mm256_storeu_pd ((*(result->data+r)+c+8), res);
            
            __m256d d4 = _mm256_loadu_pd(*(mat->data+r)+c+12);
            __m256d pos4 = _mm256_max_pd(zeroes, d4);
            __m256d neg4 = _mm256_max_pd(zeroes, _mm256_sub_pd(zeroes, d4));
            res = _mm256_add_pd(pos4, neg4);
            _mm256_storeu_pd ((*(result->data+r)+c+12), res);
        }
        
        
        
        for (int c = mat->cols/16*16; c < mat->cols/4*4; c+=4) {
            __m256d d1 = _mm256_loadu_pd(*(mat->data+r)+c);
            __m256d pos = _mm256_max_pd(zeroes, d1);
            __m256d neg = _mm256_max_pd(zeroes, _mm256_sub_pd(zeroes, d1));
            res = _mm256_add_pd(pos, neg);
            _mm256_storeu_pd ((*(result->data+r)+c), res);
        }
        
        
        for (int c = mat->cols/4*4; c<mat->cols; c++){
            int val = mat->data[r][c];
            result->data[r][c] = (val < 0)? 0-val : val;
        }
    }
    
   
    return 0;
}

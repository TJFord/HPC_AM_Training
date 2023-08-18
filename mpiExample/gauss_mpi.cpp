/*
 *  Gaussian Elimination Solver, based on a version distributed by Sandhya
 *  Dwarkadas, University of Rochester
 *  Parallel version is created by Jifu Tan and Haolin Ma 
 * 	20-10-2011
 */

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>  // this gives access to getopt()
// mpi head file
#include <mpi.h>
# include <omp.h>
/*
 *  Helper macro: swap two doubles
 */
#define DSWAP(a, b) { double tmp; tmp = a; a = b; b = tmp; }

/*
 *  Helper macro: swap two pointers
 */
#define PSWAP(a, b) { double *tmp; tmp = a; a = b; b = tmp; }

/*
 *  Helper macro: absolute value
 */
#define ABS(a)      (((a) > 0) ? (a) : -(a))

/*
 *  The 2-d matrix that holds the coefficients for gaussian elimination.  Since
 *  the size is an input parameter, we implement the matrix as an array of
 *  arrays.  This also makes swapping rows easy... we just swap pointers.
 */
double **matrix;


/*
 *  To verify our work, we'll back-substitute into M after it is in triangular
 *  form.  This vector gives us the solution.  As with 'V', we declare it early
 *  and allocate it early to avoid out-of-memory errors later.
 */
double *C;

/*
 * Allocate the arrays
 */
void allocate_memory(int size)
{
    /* hold [size] pointers*/
    matrix = (double**)malloc(size * sizeof(double*));
    assert(matrix != NULL);
	
    /* get a [size x size] array of doubles */
	double *tmp = (double*)malloc(size*(size+1)*sizeof(double)); // last element to save B
    assert(tmp != NULL);
    /* allocate parts of the array to the rows of the matrix */
    for (int i = 0; i < size; i++) {
        matrix[i] = tmp;
	    tmp = tmp + size+1; // last element to save B
    }
	
    /* allocate the solution vector */
    C = (double*)malloc(size * sizeof(double));
    assert(C != NULL);
}

/*
 * Initialize the matrix with some values that we know yield a solution that is
 * easy to verify. A correct solution should yield -0.5 and 0.5 for the first
 * and last C values, and 0 for the rest.
 */
void initMatrix(int nsize)
{
    //int num_thrd=16;
	//# pragma omp parallel for num_threads(num_thrd)
	for (int i = 0; i < nsize; i++) {
		//# pragma omp parallel for num_threads(num_thrd)
        for (int j = 0; j < nsize; j++) {
            matrix[i][j] = ((j < i )? 2*(j+1) : 2*(i+1));
        }
		// vector B is stored in the last column
		matrix[i][nsize] =(double)i;// B
    }
}


/*
 * Do back-substitution to get a solution vector
 */
void solveGauss(int nsize)
{
   
	C[nsize-1] = matrix[nsize-1][nsize];
    for (int row = nsize - 2; row >= 0; row--) {
	   C[row] =matrix[row][nsize];
        for (int col = nsize - 1; col > row; col--) {
            C[row] -= matrix[row][col] * C[col];
        }
    }
}

/*
 *  Main routine: parse command args, create array, compute solution
 */
int main(int argc, char *argv[])
{
    /* start and end time */
    struct timeval t0;
    struct timeval t1;
    /* two temps and the size of the array */
    int i, j,s, nsize =1024;
	
    /* get the size from the command-line */
    while ((i = getopt(argc,argv,"s:")) != -1) {
        switch(i) {
          case 's':
            s = atoi(optarg);
            if (s > 0)
                nsize = s;
            else
                printf("  -s is negative... using %d\n", nsize);
            break;
          default:
            assert(0);
            break;
        }
    }
	/*
	* MPI starts
	*/
	double *pivotVal; // pivot row 
		
	int my_rank, comm_sz,num_rows,mpi_err;
	double ** local_mtx;	// part of matrix allocated in each processor
	double *local_B;
	//MPI initialization
	MPI_Init(NULL,NULL);
	MPI_Comm_size(MPI_COMM_WORLD,&comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank); 
	num_rows=nsize/comm_sz;
	

	// local memory allocation
	local_mtx = (double**)malloc(nsize*sizeof(double*));
	pivotVal=(double*)malloc((nsize+1)* sizeof(double));
	// struct defined to store the pivot element and its row, used by MPI_Reduce, MPI_MAXLOC
	struct { double val;
			int index;} pivt_in,pivt_out;
			
	double *tmp = (double*)malloc(num_rows*(nsize+1)*sizeof(double));
	
    //allocate parts of the array to the rows of the matrix 
    for (int i = 0; i < num_rows; i++) {
        local_mtx[i] = tmp;
        tmp = tmp + nsize+1; // last one for B
    }
	//end of local memory allocation*/
	
	//******* data distribution based on cyclic row distribution*****//
		if (my_rank==0) 
		{// allocate memory 
			allocate_memory(nsize);
			// get start time, initialize, compute, solve, get end time 
			gettimeofday(&t0, 0);
			initMatrix(nsize);
			//int num_thrd=16;
			//# pragma omp parallel for num_threads(num_thrd)
			for (i=0;i<nsize;i++)
					{	
						if(i%comm_sz ==0)
							{
								local_mtx[(i-my_rank)/comm_sz]=matrix[i];
							}
						else //send data to other processors based on cyclic row distribution
							{
								MPI_Send(matrix[i],nsize+1,MPI_DOUBLE,i%comm_sz,0,MPI_COMM_WORLD);
							}
					}
		}
		else	// receive data from other processor 0
		{		
			for (i=0;i<num_rows;i++)
				{MPI_Recv(local_mtx[i],nsize+1,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);}
		}
	
	//******* pivoting and elimination of the matrix ********//
	
	
		for (j=0;j<nsize;j++)	//loop iindex for column
		{	
			pivt_in.val=0.0;	// strut used by MPI_MAXLOC, component val is the maximum value
			pivt_in.index=0;	//	component index is the pivot row number
			if(my_rank==j%comm_sz)		//if element matrix(j,j) is inside my rank, find the maximum inside my rank
				{
					pivt_in.val = ABS(local_mtx[(j-my_rank)/comm_sz][j]);
					pivt_in.index =j;
					for (i=(j-my_rank)/comm_sz;i<num_rows;i++)
					{
						double tmp = ABS(local_mtx[i][j]);
						if (tmp > pivt_in.val) {
							pivt_in.val = tmp;
							pivt_in.index =i*comm_sz+my_rank;
						}
					}
				}
			else		// find the maximum in other processors
				{
						pivt_in.val = 0.0;
						
						for (i=0;i<num_rows;i++)
						{	
							if(i*comm_sz+my_rank>j)
							{
								double tmp = ABS(local_mtx[i][j]);
								if (tmp > pivt_in.val) {
									pivt_in.val = tmp;
									pivt_in.index =i*comm_sz+my_rank;
								}
							}
						}
				}
		// MPI_MAXLOC is used to get the maximum value and its row number			
			MPI_Reduce(&pivt_in,&pivt_out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD); 
		// brocast the struct to every processor
			MPI_Bcast(&pivt_out,1,MPI_DOUBLE_INT,0,MPI_COMM_WORLD);

//********** get pivot row based on struct defined before, brocast to each processor.
// if the exact amount of data is broadcast, the time decreased from 17s to 13 s!
			if(my_rank==(pivt_out.index)%comm_sz) 
				{
					int k=(pivt_out.index-(pivt_out.index)%comm_sz)/comm_sz;
					//# pragma omp parallel for num_threads(16)
					for (i=0;i<nsize+1;i++)
						{pivotVal[i]=local_mtx[k][i];} // Not directly copy address!
				}
			MPI_Bcast(&pivotVal[j],nsize+1-j,MPI_DOUBLE,(pivt_out.index)%comm_sz,MPI_COMM_WORLD);
			
			// Swap the row if the pivot row is not the row where matrix(j,j) is
			if(pivt_out.index !=j)
			{
				if(my_rank==j%comm_sz)
					{
					// swap the row 
							{	
								double * row_p=pivotVal;
								int k=(j-j%comm_sz)/comm_sz;
								PSWAP(row_p,local_mtx[k]);
								MPI_Send(row_p,nsize+1,MPI_DOUBLE,pivt_out.index%comm_sz,0,MPI_COMM_WORLD);

							}
					}	
				if(my_rank==pivt_out.index%comm_sz)
				{	
					int k=(pivt_out.index-pivt_out.index%comm_sz)/comm_sz;
					MPI_Recv(local_mtx[k],nsize+1,MPI_DOUBLE,j%comm_sz,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				}	
			}
			
	//** elimination process, after the step, the matrix is completed!  **//			
			if(my_rank==(j%comm_sz)) // j row is inside my rank
				{	
					int k=(j-j%comm_sz)/comm_sz;
					double temp=local_mtx[k][j];
					local_mtx[k][j]=1.0;
					for (i=j+1;i<nsize+1;i++) 
					{
					local_mtx[k][i]/=temp;	// normalize the pivot row 
					}
					for(int m=k+1;m<num_rows;m++)
					{	
						double temp=local_mtx[m][j];
						
						for (i=j+1;i<nsize+1;i++)
							{
								// calculate the element for other rows that are in the same processor
								local_mtx[m][i]-=pivotVal[i]*temp/pivotVal[j];  

							}
							// same column is zero
						local_mtx[m][j]=0.0;
					}
				}
				else	// calculate the data in other rows in other processors
				{	for (int m=0;m<num_rows;m++)
							{	
								if((m*comm_sz+my_rank)>j)
								{
									
									double temp=local_mtx[m][j];
									
									for (int i=j+1;i<nsize+1;i++)
										{
										// calculate the element for other rows that are in the same processor
										local_mtx[m][i]-=pivotVal[i]*temp/pivotVal[j];

										}
										// same column is zero
										local_mtx[m][j]=0.0;
									
								}
							}
				}
				

		}
		
//**	send the local_mtx to processor 0 to do the back substitution	**//
		for(j=1;j<comm_sz;j++)	// processor index except 0
		{
			if(my_rank==j)
				{for (i=0;i<num_rows;i++)
					{
						MPI_Send(&local_mtx[i][i*comm_sz+j],nsize+1-(i*comm_sz+j),MPI_DOUBLE,0,0,MPI_COMM_WORLD);}
					}
		}
		if(my_rank==0)
			{	
				// receive data and keep them in matrix
				for(i=1;i<comm_sz;i++)
					{
						for (j=0;j<num_rows;j++)
						{
						MPI_Recv(&matrix[j*comm_sz+i][j*comm_sz+i],nsize+1-(j*comm_sz+i),MPI_DOUBLE,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						}
					}
					
					//compute results
					solveGauss(nsize);
					gettimeofday(&t1, 0);
					// verify that the code was correct 
					for (int n = 0; n < nsize; n++) {
						if (n == 0)
							assert (C[n] == -0.5);
						else if (n == nsize-1)
							assert (C[n] == 0.5);
						else
							assert (C[n] == 0);
					}
					printf("Correct solution found.\n");
			} 

	if(my_rank==0)
	{
    // print compute time 
    unsigned long usecs = t1.tv_usec - t0.tv_usec;
    usecs  += (1000000 * (t1.tv_sec - t0.tv_sec));
    printf("Size: %d rows\n", nsize);
    printf("Time: %f seconds\n", ((double)usecs)/1000000.0);}
    
	MPI_Finalize();
	//printf("end\n");
	return 0;
}

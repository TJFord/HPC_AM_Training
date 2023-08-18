# the name of the executable.  Of course, this is also the base of the source
# file name
TARGET   = gauss_mpi

# define the configuration and build flags and rules
CC       = mpiCC
CFLAGS   = -O3 -Wall -fopenmp
LDFLAGS  = -lm

# transform target names into the names of C files
SOURCES  = $(patsubst %, %.cpp,  ${TARGET})

# Tell make that some rules don't actually produce what it thinks they produce
.PHONY: all clean run done config

# default rule builds the executable named by TARGET
all: $(TARGET)

# how to build the target
$(TARGET): ${SOURCES}
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# cleanup by removing the output of the build step
clean:
	-rm -rf $(TARGET)

# this makes life much easier: 'gmake config' will configure MPI
config: hostfile
	recon hostfile
	lamboot -v hostfile

# another nice trick: 'gmake run' will run our program a few ways, as
# specified in the following lines
run: $(TARGET) hostfile
	@echo "Running 16 CPU"
	mpirun n1-16 -ssi rpi usysv $(TARGET) -s 4096
	@echo "Running 16 nodes"
	mpirun n1-16 -ssi rpi usysv $(TARGET) -s 1024
	@echo "Running 16 nodes"
	mpirun n1-16 -ssi rpi usysv $(TARGET) -s 512
	@echo "Running 8 CPU"
	mpirun n1-8 -ssi rpi usysv $(TARGET) -s 4096
	@echo "Running 8 nodes"
	mpirun n1-8 -ssi rpi usysv $(TARGET) -s 1024
	@echo "Running 8 nodes"
	mpirun n1-8 -ssi rpi usysv $(TARGET) -s 512
	@echo "Running 4 CPU"
	mpirun n1-4 -ssi rpi usysv $(TARGET) -s 4096
	@echo "Running 4 nodes"
	mpirun n1-4 -ssi rpi usysv $(TARGET) -s 1024
	@echo "Running 4 nodes"
	mpirun n1-4 -ssi rpi usysv $(TARGET) -s 512


# when we're done, we need to shut down MPI
done: hostfile
	lamhalt hostfile

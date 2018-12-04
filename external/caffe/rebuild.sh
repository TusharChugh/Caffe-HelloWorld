#!/bin/bash

echo "Removing current build directory"
rm -rf build/
rm -rf include/caffe/proto

echo "Running CMAKE"
mkdir -p build/
cd build/
cmake -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF ..

# Get core count

NPROC_EXISTS=`which nproc | wc -l`

cores_to_use=4
if [ $NPROC_EXISTS ]; then
    core_count=`nproc`
    cores_to_use=`expr $core_count - 1`
fi

echo "Building with ${cores_to_use} threads"
echo ""
echo ""
make -j${cores_to_use}
sudo make install

echo ""
echo "Generating protoc files"
echo ""

cd ..
protoc src/caffe/proto/caffe.proto --cpp_out=.
mkdir include/caffe/proto
mv src/caffe/proto/caffe.pb.h include/caffe/proto

echo ""
echo "Finished rebuilding caffe."
echo ""

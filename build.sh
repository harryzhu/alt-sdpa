#! /bin/bash

rm -rf ./build/*

if [ ! -d "build" ];then
	mkdir build
fi

cmake -B build
cd build
make

rm -rf ./alt_sdpa/*

if [ ! -d "alt_sdpa" ];then
	mkdir alt_sdpa
fi

cd ..

find ./build -type f -name "alt_sdpa_ext.*.so" -exec cp {} ./alt_sdpa \;
cp ./src/alt_sdpa_wrapper.py  ./alt_sdpa/
cp ./src/__init__.py  ./alt_sdpa/

#!/bin/sh

DIST=3rdparty
ARCH=ubuntu1804_tx2

mkdir -p $ARCH/lib
mkdir -p $ARCH/include
mkdir -p $ARCH/bin

echo Copying archives...
cp -f samples/15_zznvcodec/*.so $ARCH/lib

echo Copying header files...
cp -f samples/15_zznvcodec/zznvcodec.h $ARCH/include

echo Copying executable files...
cp -f samples/15_zznvcodec/test_zznvdec $ARCH/bin
cp -r samples/15_zznvcodec/test_zznvenc $ARCH/bin

echo Modifying file attributes...
chmod +rw $ARCH/lib/*
chmod +rw $ARCH/include/*
chmod +rw $ARCH/bin/*

echo Done.

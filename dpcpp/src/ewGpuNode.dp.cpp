/*
 * EasyWave - A realtime tsunami simulation program with GPU support.
 * Copyright (C) 2014  Andrey Babeyko, Johannes Spazier
 * GFZ German Research Centre for Geosciences (http://www.gfz-potsdam.de)
 *
 * Parts of this program (especially the GPU extension) were developed
 * within the context of the following publicly funded project:
 * - TRIDEC, EU 7th Framework Programme, Grant Agreement 258723
 *   (http://www.tridec-online.eu)
 *
 * Licensed under the EUPL, Version 1.1 or - as soon they will be approved by
 * the European Commission - subsequent versions of the EUPL (the "Licence"),
 * complemented with the following provision: For the scientific transparency
 * and verification of results obtained and communicated to the public after
 * using a modified version of the work, You (as the recipient of the source
 * code and author of this modified version, used to produce the published
 * results in scientific communications) commit to make this modified source
 * code available in a repository that is easily and freely accessible for a
 * duration of five years after the communication of the obtained results.
 * 
 * You may not use this work except in compliance with the Licence.
 * 
 * You may obtain a copy of the Licence at:
 * https://joinup.ec.europa.eu/software/page/eupl
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the Licence is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the Licence for the specific language governing permissions and
 * limitations under the Licence.
 */

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "ewGpuNode.dp.hpp"
#include "ewCudaKernels.dp.hpp"
#include <cmath>

#include <algorithm>

#include <chrono>

CGpuNode::CGpuNode() {

	pitch = 0;
	copied = true;

	for( int i = 0; i < 5; i++ ) {
                /*
                DPCT1026:0: The call to cudaEventCreate was removed, because
                this call is redundant in DPC++.
                */
                /*
                DPCT1026:1: The call to cudaEventCreate was removed, because
                this call is redundant in DPC++.
                */
                dur[i] = 0.0;
        }

}

int CGpuNode::mallocMem() try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

        CArrayNode::mallocMem();

	Params& dp = data.params;

	/* fill in some fields here */
	dp.nI = NLon;
	dp.nJ = NLat;
	dp.sshArrivalThreshold = Par.sshArrivalThreshold;
	dp.sshClipThreshold = Par.sshClipThreshold;
	dp.sshZeroThreshold = Par.sshZeroThreshold;
	dp.lpad = 31;

	size_t nJ_aligned = dp.nJ + dp.lpad;

	/* 2-dim */
        /*
        DPCT1003:2: Migrated API does not return error code. (*, 0) is inserted.
        You may need to rewrite this code.
        */
        data.d = (float *)dpct::dpct_malloc(pitch, nJ_aligned * sizeof(float), dp.nI);
        /*
        DPCT1003:5: Migrated API does not return error code. (*, 0) is inserted.
        You may need to rewrite this code.
        */
        data.h = (float *)dpct::dpct_malloc(pitch, nJ_aligned * sizeof(float), dp.nI);
        /*
        DPCT1003:6: Migrated API does not return error code. (*, 0) is inserted.
        You may need to rewrite this code.
        */
        data.hMax = (float *)dpct::dpct_malloc(pitch, nJ_aligned * sizeof(float), dp.nI);
        /*
        DPCT1003:7: Migrated API does not return error code. (*, 0) is inserted.
        You may need to rewrite this code.
        */
        data.fM = (float *)dpct::dpct_malloc(pitch, nJ_aligned * sizeof(float), dp.nI);
        /*
        DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted.
        You may need to rewrite this code.
        */
        data.fN = (float *)dpct::dpct_malloc(pitch, nJ_aligned * sizeof(float), dp.nI);
        /*
        DPCT1003:9: Migrated API does not return error code. (*, 0) is inserted.
        You may need to rewrite this code.
        */
        data.cR1 = (float *)dpct::dpct_malloc(pitch, nJ_aligned * sizeof(float), dp.nI);
        /*
        DPCT1003:10: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        data.cR2 = (float *)dpct::dpct_malloc(pitch, nJ_aligned * sizeof(float), dp.nI);
        /*
        DPCT1003:11: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        data.cR4 = (float *)dpct::dpct_malloc(pitch, nJ_aligned * sizeof(float), dp.nI);
        /*
        DPCT1003:12: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        data.tArr = (float *)dpct::dpct_malloc(pitch, nJ_aligned * sizeof(float), dp.nI);
        /* TODO: cR3, cR5 for coriolis */

	/* 1-dim */
        /*
        DPCT1003:13: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        data.cR6 = sycl::malloc_device<float>(dp.nJ, q_ct1);
        /*
        DPCT1003:14: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        (data.cB1) = sycl::malloc_device<float>(dp.nI, q_ct1);
        /*
        DPCT1003:15: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        data.cB2 = sycl::malloc_device<float>(dp.nJ, q_ct1);
        /*
        DPCT1003:16: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        data.cB3 = sycl::malloc_device<float>(dp.nI, q_ct1);
        /*
        DPCT1003:17: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        data.cB4 = sycl::malloc_device<float>(dp.nJ, q_ct1);

        /*
        DPCT1003:18: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        data.g_MinMax = sycl::malloc_device<sycl::int4>(1, q_ct1);

        /* TODO: make sure that pitch is a multiple of 4 and the same for each cudaMallocPitch() call */
	dp.pI = pitch / sizeof(float);

	return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int CGpuNode::copyToGPU() try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

        Params& dp = data.params;

	/* align left grid boundary to a multiple of 32 with an offset 1 */
        Jmin -= (Jmin-2) % 32;
       
        /* fill in further fields here */
        dp.iMin = Imin;
	dp.iMax = Imax;
        dp.jMin = Jmin;
	dp.jMax = Jmax;

	/* add offset to data.d to guarantee alignment: data.d + LPAD */
	/* 2-dim */
        /*
        DPCT1003:19: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        dpct::dpct_memcpy(
                       data.d + dp.lpad, pitch, d, dp.nJ * sizeof(float),
                       dp.nJ * sizeof(float), dp.nI, dpct::host_to_device);
        /*
        DPCT1003:20: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        dpct::dpct_memcpy(
                       data.h + dp.lpad, pitch, h, dp.nJ * sizeof(float),
                       dp.nJ * sizeof(float), dp.nI, dpct::host_to_device);
        /*
        DPCT1003:21: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        dpct::dpct_memcpy(
                       data.hMax + dp.lpad, pitch, hMax, dp.nJ * sizeof(float),
                       dp.nJ * sizeof(float), dp.nI, dpct::host_to_device);
        /*
        DPCT1003:22: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        dpct::dpct_memcpy(
                       data.fM + dp.lpad, pitch, fM, dp.nJ * sizeof(float),
                       dp.nJ * sizeof(float), dp.nI, dpct::host_to_device);
        /*
        DPCT1003:23: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        dpct::dpct_memcpy(
                       data.fN + dp.lpad, pitch, fN, dp.nJ * sizeof(float),
                       dp.nJ * sizeof(float), dp.nI, dpct::host_to_device);
        /*
        DPCT1003:24: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        dpct::dpct_memcpy(
                       data.cR1 + dp.lpad, pitch, cR1, dp.nJ * sizeof(float),
                       dp.nJ * sizeof(float), dp.nI, dpct::host_to_device);
        /*
        DPCT1003:25: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        dpct::dpct_memcpy(
                       data.cR2 + dp.lpad, pitch, cR2, dp.nJ * sizeof(float),
                       dp.nJ * sizeof(float), dp.nI, dpct::host_to_device);
        /*
        DPCT1003:26: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        dpct::dpct_memcpy(
                       data.cR4 + dp.lpad, pitch, cR4, dp.nJ * sizeof(float),
                       dp.nJ * sizeof(float), dp.nI, dpct::host_to_device);
        /*
        DPCT1003:27: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        dpct::dpct_memcpy(
                       data.tArr + dp.lpad, pitch, tArr, dp.nJ * sizeof(float),
                       dp.nJ * sizeof(float), dp.nI, dpct::host_to_device);

        /* FIXME: move global variables into data structure */
	/* 1-dim */
        /*
        DPCT1003:28: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        q_ct1.memcpy(data.cR6, R6, dp.nJ * sizeof(float)).wait();
        /*
        DPCT1003:29: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        q_ct1.memcpy(data.cB1, C1, dp.nI * sizeof(float)).wait();
        /*
        DPCT1003:30: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        q_ct1.memcpy(data.cB2, C2, dp.nJ * sizeof(float)).wait();
        /*
        DPCT1003:31: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        q_ct1.memcpy(data.cB3, C3, dp.nI * sizeof(float)).wait();
        /*
        DPCT1003:32: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        q_ct1.memcpy(data.cB4, C4, dp.nJ * sizeof(float)).wait();

        return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
int CGpuNode::copyFromGPU() try {

        Params& dp = data.params;

        /*
        DPCT1003:33: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        dpct::dpct_memcpy(
                       hMax, dp.nJ * sizeof(float), data.hMax + dp.lpad, pitch,
                       dp.nJ * sizeof(float), dp.nI, dpct::device_to_host);
        /*
        DPCT1003:34: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        dpct::dpct_memcpy(
                       tArr, dp.nJ * sizeof(float), data.tArr + dp.lpad, pitch,
                       dp.nJ * sizeof(float), dp.nI, dpct::device_to_host);

        return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int CGpuNode::copyIntermediate() try {

        /* ignore copy requests if data already present on CPU side */
	if( copied )
		return 0;

	Params& dp = data.params;

        /*
        DPCT1003:35: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        dpct::dpct_memcpy(h, dp.nJ * sizeof(float), data.h + dp.lpad,
                                     pitch, dp.nJ * sizeof(float), dp.nI,
                                     dpct::device_to_host);

        /* copy finished */
	copied = true;

	return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int CGpuNode::copyPOIs() try {

        Params& dp = data.params;

	if( copied )
		return 0;

	for( int n = 0; n < NPOIs; n++ ) {

		int i = idxPOI[n] / dp.nJ + 1;
		int j = idxPOI[n] % dp.nJ + 1;

		int id = data.idx( i, j );

                /*
                DPCT1003:36: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                dpct::get_default_queue()
                               .memcpy(h + idxPOI[n], data.h + dp.lpad + id,
                                       sizeof(float))
                               .wait();
        }

	return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int CGpuNode::freeMem() try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

        /* 2-dim */
        /*
        DPCT1003:37: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        sycl::free(data.d, q_ct1);
        /*
        DPCT1003:38: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        sycl::free(data.h, q_ct1);
        /*
        DPCT1003:39: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        sycl::free(data.hMax, q_ct1);
        /*
        DPCT1003:40: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        sycl::free(data.fM, q_ct1);
        /*
        DPCT1003:41: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        sycl::free(data.fN, q_ct1);
        /*
        DPCT1003:42: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        sycl::free(data.cR1, q_ct1);
        /*
        DPCT1003:43: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        sycl::free(data.cR2, q_ct1);
        /*
        DPCT1003:44: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        sycl::free(data.cR4, q_ct1);
        /*
        DPCT1003:45: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        sycl::free(data.tArr, q_ct1);

        /* 1-dim */
        /*
        DPCT1003:46: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        sycl::free(data.cR6, q_ct1);
        /*
        DPCT1003:47: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        sycl::free(data.cB1, q_ct1);
        /*
        DPCT1003:48: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        sycl::free(data.cB2, q_ct1);
        /*
        DPCT1003:49: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        sycl::free(data.cB3, q_ct1);
        /*
        DPCT1003:50: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        sycl::free(data.cB4, q_ct1);

        /*
        DPCT1003:51: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        sycl::free(data.g_MinMax, q_ct1);

        float total_dur = 0.f;
	for( int j = 0; j < 5; j++ ) {
		printf_v("Duration %u: %.3f\n", j, dur[j]);
		total_dur += dur[j];
	}
	printf_v("Duration total: %.3f\n",total_dur);

	CArrayNode::freeMem();

	return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int CGpuNode::run() try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

        Params& dp = data.params;

	int nThreads = 256;
	int xThreads = 32;
	int yThreads = nThreads / xThreads;

	int NJ = dp.jMax - dp.jMin + 1;
	int NI = dp.iMax - dp.iMin + 1;
	int xBlocks = ceil( (float)NJ / (float)xThreads );
	int yBlocks = ceil( (float)NI / (float)yThreads );

        sycl::range<3> threads(1, yThreads, xThreads);
        sycl::range<3> blocks(1, yBlocks, xBlocks);

        int nBlocks = ceil((float)std::max(dp.nI, dp.nJ) / (float)nThreads);

        dp.mTime = Par.time;

        /*
        DPCT1012:54: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:55: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        evtStart_ct1[0] = std::chrono::steady_clock::now();
        CUDA_CALL(0);
        /*
        DPCT1049:52: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
                auto data_ct0 = data;

                cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                         runWaveUpdateKernel(data_ct0);
                                 });
        });
        /*
        DPCT1012:56: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:57: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        evtEnd_ct1[0] = std::chrono::steady_clock::now();
        CUDA_CALL(0);
        /*
        DPCT1012:58: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:59: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        evtStart_ct1[1] = std::chrono::steady_clock::now();
        CUDA_CALL(0);
        /*
        DPCT1049:60: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
                auto data_ct0 = data;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, nBlocks) *
                                          sycl::range<3>(1, 1, nThreads),
                                      sycl::range<3>(1, 1, nThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                            runWaveBoundaryKernel(data_ct0);
                    });
        });
        /*
        DPCT1012:61: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:62: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        evtEnd_ct1[1] = std::chrono::steady_clock::now();
        CUDA_CALL(0);
        /*
        DPCT1012:63: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:64: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        evtStart_ct1[2] = std::chrono::steady_clock::now();
        CUDA_CALL(0);
        /*
        DPCT1049:53: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
                auto data_ct0 = data;

                cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                         runFluxUpdateKernel(data_ct0);
                                 });
        });
        /*
        DPCT1012:65: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:66: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        evtEnd_ct1[2] = std::chrono::steady_clock::now();
        CUDA_CALL(0);
        /*
        DPCT1012:67: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:68: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        evtStart_ct1[3] = std::chrono::steady_clock::now();
        CUDA_CALL(0);
        /*
        DPCT1049:69: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
                auto data_ct0 = data;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, nBlocks) *
                                          sycl::range<3>(1, 1, nThreads),
                                      sycl::range<3>(1, 1, nThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                            runFluxBoundaryKernel(data_ct0);
                    });
        });
        /*
        DPCT1012:70: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:71: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        evtEnd_ct1[3] = std::chrono::steady_clock::now();
        CUDA_CALL(0);
        /*
        DPCT1012:72: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:73: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        evtStart_ct1[4] = std::chrono::steady_clock::now();
        CUDA_CALL(0);
        /*
        DPCT1003:74: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        q_ct1.memset(data.g_MinMax, 0, sizeof(sycl::int4)).wait();
        /*
        DPCT1049:75: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
                auto data_ct0 = data;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, nBlocks) *
                                          sycl::range<3>(1, 1, nThreads),
                                      sycl::range<3>(1, 1, nThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                            runGridExtendKernel(data_ct0);
                    });
        });
        /*
        DPCT1012:76: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:77: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        evtEnd_ct1[4] = std::chrono::steady_clock::now();
        CUDA_CALL(0);

        sycl::int4 MinMax;
        /*
        DPCT1003:78: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        q_ct1.memcpy(&MinMax, data.g_MinMax, sizeof(sycl::int4)).wait();
        dev_ct1.queues_wait_and_throw();

        if (MinMax.x())
            Imin = dp.iMin = std::max(dp.iMin - 1, 2);

        if (MinMax.y())
            Imax = dp.iMax = std::min(dp.iMax + 1, dp.nI - 1);

        if (MinMax.z())
            Jmin = dp.jMin = std::max(dp.jMin - 32, 2);

        if (MinMax.w())
            Jmax = dp.jMax = std::min(dp.jMax + 1, dp.nJ - 1);

        float _dur;
	for( int j = 0; j < 5; j++ ) {
                _dur = std::chrono::duration<float, std::milli>(evtEnd_ct1[j] -
                                                                evtStart_ct1[j])
                           .count();
                dur[j] += _dur;
	}

	/* data has changed now -> copy becomes necessary */
	copied = false;

	return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

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

void runWaveUpdateKernel( KernelData data , sycl::nd_item<3> item_ct1) {

  Params& dp = data.params;

  int i = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
          item_ct1.get_local_id(1) + dp.iMin;
  int j = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
          item_ct1.get_local_id(2) + dp.jMin;
  int ij = data.idx(i,j);
  float absH;

  /* maybe unnecessary if controlled from outside */
  if( i <= dp.iMax && j <= dp.jMax && data.d[ij] != 0 ) {

	  float hh = data.h[ij] - data.cR1[ij] * ( data.fM[ij] - data.fM[data.le(ij)] + data.fN[ij] * data.cR6[j] - data.fN[data.dn(ij)]*data.cR6[j-1] );

          absH = sycl::fabs(hh);

          if( absH < dp.sshZeroThreshold ) {
		  hh = 0.f;
	  } else if( hh > data.hMax[ij] ) {
		  data.hMax[ij] = hh;
		  //hMax[ij] = fmaxf(hMax[ij],h[ij]);
	  }

	  if( dp.sshArrivalThreshold && data.tArr[ij] < 0 && absH > dp.sshArrivalThreshold )
	  	  data.tArr[ij] = dp.mTime;

	  data.h[ij] = hh;
  }

}

void runFluxUpdateKernel( KernelData data , sycl::nd_item<3> item_ct1) {

	Params& dp = data.params;

        int i = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
                item_ct1.get_local_id(1) + dp.iMin;
        int j = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                item_ct1.get_local_id(2) + dp.jMin;
        int ij = data.idx(i,j);

	if( i <= dp.iMax && j <= dp.jMax && data.d[ij] != 0 ) {

	  float hh = data.h[ij];

	  if( data.d[data.ri(ij)] != 0 ) {
		  data.fM[ij] = data.fM[ij] - data.cR2[ij]*(data.h[data.ri(ij)] - hh);
	  }

	  if( data.d[data.up(ij)] != 0 )
		  data.fN[ij] = data.fN[ij] - data.cR4[ij]*(data.h[data.up(ij)] - hh);

	}

}

#define SQR(x)   ((x) * (x))

void runWaveBoundaryKernel( KernelData data , sycl::nd_item<3> item_ct1) {

	KernelData& dt = data;
	Params& dp = data.params;

        int id = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                 item_ct1.get_local_id(2) + 2;
        int ij;

	if( id <= dp.nI-1 ) {
	  ij = dt.idx(id,1);
          dt.h[ij] = sycl::sqrt(SQR(dt.fN[ij]) +
                                0.25f * SQR((dt.fM[ij] + dt.fM[dt.le(ij)]))) *
                     dt.cB1[id - 1];
          if( dt.fN[ij] > 0 ) dt.h[ij] = -dt.h[ij];
	}

	if( id <= dp.nI-1 ) {
	  ij = dt.idx(id,dp.nJ);
          dt.h[ij] = sycl::sqrt(SQR(dt.fN[dt.dn(ij)]) +
                                0.25f * SQR((dt.fM[ij] + dt.fM[dt.dn(ij)]))) *
                     dt.cB3[id - 1];
          if( dt.fN[dt.dn(ij)] < 0 ) dt.h[ij] = -dt.h[ij];
	}

	if( id <= dp.nJ-1 ) {
	  ij = dt.idx(1,id);
          dt.h[ij] = sycl::sqrt(SQR(dt.fM[ij]) +
                                0.25f * SQR((dt.fN[ij] + dt.fN[dt.dn(ij)]))) *
                     dt.cB2[id - 1];
          if( dt.fM[ij] > 0 ) dt.h[ij] = -dt.h[ij];
	}

	if( id <= dp.nJ-1 ) {
	  ij = dt.idx(dp.nI,id);
          dt.h[ij] = sycl::sqrt(SQR(dt.fM[dt.le(ij)]) +
                                0.25f * SQR((dt.fN[ij] + dt.fN[dt.dn(ij)]))) *
                     dt.cB4[id - 1];
          if( dt.fM[dt.le(ij)] < 0 ) dt.h[ij] = -dt.h[ij];
	}

	if( id == 2 ) {
	  ij = dt.idx(1,1);
          dt.h[ij] = sycl::sqrt(SQR(dt.fM[ij]) + SQR(dt.fN[ij])) * dt.cB1[0];
          if( dt.fN[ij] > 0 ) dt.h[ij] = -dt.h[ij];

	  ij = dt.idx(dp.nI,1);
          dt.h[ij] = sycl::sqrt(SQR(dt.fM[dt.le(ij)]) + SQR(dt.fN[ij])) *
                     dt.cB1[dp.nI - 1];
          if( dt.fN[ij] > 0 ) dt.h[ij] = -dt.h[ij];

	  ij = dt.idx(1,dp.nJ);
          dt.h[ij] = sycl::sqrt(SQR(dt.fM[ij]) + SQR(dt.fN[dt.dn(ij)])) * dt.cB3[0];
          if( dt.fN[dt.dn(ij)] < 0 ) dt.h[ij] = -dt.h[ij];

	  ij = dt.idx(dp.nI,dp.nJ);
          dt.h[ij] = sycl::sqrt(SQR(dt.fM[dt.le(ij)]) + SQR(dt.fN[dt.dn(ij)])) *
                     dt.cB3[dp.nI - 1];
          if( dt.fN[dt.dn(ij)] < 0 ) dt.h[ij] = -dt.h[ij];
	}
}

void runFluxBoundaryKernel( KernelData data , sycl::nd_item<3> item_ct1) {

	KernelData& dt = data;
	Params& dp = data.params;

        int id = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                 item_ct1.get_local_id(2) + 1;
        int ij;

	if( id <= dp.nI-1 ) {
	  ij = dt.idx(id,1);
	  dt.fM[ij] = dt.fM[ij] - dt.cR2[ij]*(dt.h[dt.ri(ij)] - dt.h[ij]);
	}

	if( id <= dp.nJ ) {
	  ij = dt.idx(1,id);
	  dt.fM[ij] = dt.fM[ij] - dt.cR2[ij]*(dt.h[dt.ri(ij)] - dt.h[ij]);
	}

	if( id <= dp.nI-1 ) {
	  ij = dt.idx(id,dp.nJ);
	  dt.fM[ij] = dt.fM[ij] - dt.cR2[ij]*(dt.h[dt.ri(ij)] - dt.h[ij]);
	}

	if( id <= dp.nJ-1 ) {
	  ij = dt.idx(1,id);
	  dt.fN[ij] = dt.fN[ij] - dt.cR4[ij]*(dt.h[dt.up(ij)] - dt.h[ij]);
	}

	if( id <= dp.nI ) {
	  ij = dt.idx(id,1);
	  dt.fN[ij] = dt.fN[ij] - dt.cR4[ij]*(dt.h[dt.up(ij)] - dt.h[ij]);
	}

	if( id <= dp.nJ-1 ) {
	  ij = dt.idx(dp.nI,id);
	  dt.fN[ij] = dt.fN[ij] - dt.cR4[ij]*(dt.h[dt.up(ij)] - dt.h[ij]);
	}

}

void runGridExtendKernel( KernelData data , sycl::nd_item<3> item_ct1) {

	Params& dp = data.params;

        int id = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                 item_ct1.get_local_id(2) + 1;

#if (DPCT_COMPATIBILITY_TEMP >= 130)

        if( id >= dp.jMin && id <= dp.jMax ) {

          if (sycl::fabs(data.h[data.idx(dp.iMin + 2, id)]) > dp.sshClipThreshold)
                  sycl::atomic<int>(sycl::global_ptr<int>(&(data.g_MinMax->x)))
                      .fetch_add(1);

          if (sycl::fabs(data.h[data.idx(dp.iMax - 2, id)]) > dp.sshClipThreshold)
                  sycl::atomic<int>(sycl::global_ptr<int>(&(data.g_MinMax->y)))
                      .fetch_add(1);
        }

	if( id >= dp.iMin && id <= dp.iMax ) {

          if (sycl::fabs(data.h[data.idx(id, dp.jMin + 2)]) > dp.sshClipThreshold)
                  sycl::atomic<int>(sycl::global_ptr<int>(&(data.g_MinMax->z)))
                      .fetch_add(1);

          if (sycl::fabs(data.h[data.idx(id, dp.jMax - 2)]) > dp.sshClipThreshold)
                  sycl::atomic<int>(sycl::global_ptr<int>(&(data.g_MinMax->w)))
                      .fetch_add(1);
        }

#else

        if( id == 1 ) {

          for( int j = dp.jMin; j <= dp.jMax; j++ ) {
            
            if( fabsf(data.h[data.idx(dp.iMin+2,j)]) > dp.sshClipThreshold ) {
                data.g_MinMax->x = 1;
                break;
            }

          }

          for( int j = dp.jMin; j <= dp.jMax; j++ ) {

            if( fabsf(data.h[data.idx(dp.iMax-2,j)]) > dp.sshClipThreshold ) {
               data.g_MinMax->y = 1;
               break;
            }

          }

          for( int i = dp.iMin; i <= dp.iMax; i++ ) {
        
            if( fabsf(data.h[data.idx(i,dp.jMin+2)]) > dp.sshClipThreshold ) {
              data.g_MinMax->z = 1;
              break;
            }

          }

          for( int i = dp.iMin; i <= dp.iMax; i++ ) {

            if( fabsf(data.h[data.idx(i,dp.jMax-2)]) > dp.sshClipThreshold ) {
              data.g_MinMax->w = 1;
              break;
            }

          }

        }

#endif

}


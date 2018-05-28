#pragma once
#include "cuda_error.h"

namespace cudalib
{
    namespace deleters
    {
        /**
         * \brief cudaFree functor
         */
        struct cuda_free
        {
            void operator()(void* ptr) const noexcept
            {
                const auto status = cudaFree(ptr);
            }
        };
    }

    namespace allocators
    {
        /**
         * \brief cudaMalloc functor.
         * \param size: Requested allocation size in bytes.
         */
        struct cuda_malloc
        {
            void* operator()(size_t size) const
            {
                void* ptr = nullptr;
                const auto status = cudaMalloc(static_cast<void**>(&ptr), size);
                throw_if_cuda_error(status, "Error in cudaMalloc");
                return ptr;
            }
        };


        /**
         * \brief cudaMallocPitch functor
         * \param pitch  - Pitch for allocation
         * \param width  - Requested pitched allocation width (in bytes)
         * \param height - Requested pitched allocation height
         */
        struct cuda_malloc_pitch
        {
            void* operator()(size_t* pitch, size_t width, size_t height) const
            {
                void* ptr = nullptr;
                const auto status = cudaMallocPitch(static_cast<void**>(&ptr), pitch, width, height);
                throw_if_cuda_error(status, "Error in cudaMallocPitch");
                return ptr;
            }
        };
    }
}

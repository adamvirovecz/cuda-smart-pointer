#pragma once

#include <memory>
#include <functional>
#include "cuda_memory.h"

namespace cudalib
{
    /**
     * \brief Allocate sizeof(T) memory on the device (using the given allocator) and return
     *  memory pointer as a std::unique_ptr. The allocator may throw an exception if the CUDA
     *  API call returned an error.
     * \tparam T Type of the value to be allocated
     * \tparam Allocator Allocator functor
     * \tparam Deleter Deleter functor (to be called by std::unique_ptr)
     * \return A unique pointer to the allocated memory space.
     * \throws cuda_error: the exception is thrown if the allocation returns with an error code
     */
    template <typename T,
              typename Allocator,
              typename Deleter=cudalib::deleters::cuda_free,
              typename = std::enable_if_t<! std::is_array<T>::value>>
    std::unique_ptr<T, Deleter> make_unique()
    {
        static_assert(std::is_trivially_constructible<T>::value,
            "Only trivially constructible types are supported on the device.");
        auto dev_ptr = Allocator()(sizeof(T));
        return std::unique_ptr<T, Deleter>(static_cast<T*>(dev_ptr));
    }

    //template <typename T, typename Allocator, typename Deleter = cudalib::deleters::cuda_free, typename = typename std::enable_if<std::is_array<T>::value>::type>
    //std::unique_ptr<T, Deleter> make_unique(size_t size)
    //{
    //	using Element = typename std::remove_extent<T>::type;
    //	static_assert(std::is_trivially_constructible<Element>::value,
    //		"Only trivially constructible types are supported on the device.");
    //	auto dev_ptr = Allocator()(sizeof(Element) * size);
    //	return std::unique_ptr<T, Deleter>(static_cast<Element*>(dev_ptr));
    //}

    /**
     * \brief Allocate memory array on the device with the given allocator and return the memory
     * pointer as a std::unique_ptr. The allocator may accept arbitrary number of parameters.
     * \tparam T Type of the array to be allocated (must be array type).
     * \tparam Allocator Allocator functor. It can have arbitrary number of parameters which are
     * given as the arguments to this function.
     * \tparam Deleter Deleter functor (to be called by std::unique_ptr)
     * \tparam AllocatorArgs Allocator argument types (variadic)
     * \param allocator_args Allocator arguments. The number of arguments given must match the
     * number of arguments accepted by the allocator functor.
     * \return A std::unique_ptr to the allocated memory space.
     * \throws cuda_error: the exception is thrown if the allocation returns with an error code
     */
    template <typename T,
              typename Allocator,
              typename Deleter = cudalib::deleters::cuda_free,
              typename... AllocatorArgs,
              typename = std::enable_if_t<std::is_array<T>::value>>
    std::unique_ptr<T, Deleter> make_unique(AllocatorArgs&&... allocator_args)
    {
        using Element = typename std::remove_extent<T>::type;
        static_assert(std::is_trivially_constructible<Element>::value,
            "Only trivially constructible types are supported on the device.");
        auto dev_ptr = Allocator()(std::forward<AllocatorArgs>(allocator_args)...);
        return std::unique_ptr<T, Deleter>(static_cast<Element*>(dev_ptr));
    }
}

#pragma once
#include <stdexcept>
#include <cuda_runtime.h>

namespace cudalib
{
	class cuda_error : public std::runtime_error
	{
		cudaError error_code_;

	public:
		cuda_error(cudaError error_code, const std::string& message);
		cuda_error(cudaError error_code, const char* message);
		cuda_error(cudaError error_code);

		cudaError code() const;
	};

	inline std::string get_error_string(cudaError error_code)
	{
		return cudaGetErrorString(error_code);
	}

	inline void throw_if_cuda_error(cudaError error_code, const std::string& error_message)
	{
		if (error_code != cudaSuccess)
		{
			throw cuda_error(error_code, error_message);
		}
	}

	inline cudaError clear_cuda_error()
	{
		return cudaGetLastError();
	}

	inline cudaError get_cuda_error()
	{
		return cudaPeekAtLastError();
	}
}

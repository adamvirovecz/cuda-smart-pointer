#include "cuda_error.h"

namespace cudalib
{
	cuda_error::cuda_error(cudaError error_code, const std::string& message) :
		runtime_error(message + ": " + get_error_string(error_code)),
		error_code_(error_code)
	{
	}

	cuda_error::cuda_error(cudaError error_code, const char* message) :
		runtime_error(std::string(message) + ": " + get_error_string(error_code)),
		error_code_(error_code)
	{
	}

	cuda_error::cuda_error(cudaError error_code) : runtime_error(get_error_string(error_code)),
	                                                      error_code_(error_code)
	{
	}

	cudaError cuda_error::code() const
	{
		return error_code_;
	}
}

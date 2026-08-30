#ifndef CUDA_BUFFER_H
#define CUDA_BUFFER_H

#include <cuda_runtime.h>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

// Owns a contiguous allocation in CUDA device memory.
template <typename T>
class CudaBuffer {
public:
    CudaBuffer() = default;

    explicit CudaBuffer(std::size_t count) {
        allocate(count);
    }

    // Device allocations have unique ownership, so copying is not allowed.
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // Moving transfers the device allocation and leaves the source empty.
    CudaBuffer(CudaBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)),
          count_(std::exchange(other.count_, 0)) {}

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = std::exchange(other.ptr_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }

        return *this;
    }

    ~CudaBuffer() noexcept {
        release();
    }

    void allocate(std::size_t count) {
        // A zero-sized buffer has no device allocation.
        if (count == 0) {
            release();
            return;
        }

        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::length_error("CUDA buffer size overflows size_t");
        }

        // Allocate first so a failed reallocation leaves the current buffer intact.
        T* new_ptr = nullptr;
        cudaError_t error = cudaMalloc(&new_ptr, count * sizeof(T));

        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }

        release();
        ptr_ = new_ptr;
        count_ = count;
    }

    // Fills every byte in the allocation with the low eight bits of byteValue.
    // This mirrors cudaMemset; it does not assign a value to each T element.
    void fillBytes(int byteValue) {
        if (count_ == 0)
            return;

        cudaError_t error = cudaMemset(ptr_, byteValue, count_ * sizeof(T));

        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
    }

    void copyFromHostToDevice(const T* source, std::size_t count) {
        if (count > count_)
            throw std::out_of_range("Source is larger than CUDA buffer");

        if (source == nullptr && count != 0)
            throw std::invalid_argument("Source cannot be null");

        cudaError_t error = cudaMemcpy(
            ptr_,
            source,
            count * sizeof(T),
            cudaMemcpyHostToDevice
        );

        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
    }

    void copyFromDeviceToHost(T* destination, std::size_t count) const {
        if (count > count_)
            throw std::out_of_range("Copy exceeds CUDA buffer size");

        if (destination == nullptr && count != 0)
            throw std::invalid_argument("Destination cannot be null");

        cudaError_t error = cudaMemcpy(
            destination,
            ptr_,
            count * sizeof(T),
            cudaMemcpyDeviceToHost
        );

        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
    }

    T* data() noexcept {
        return ptr_;
    }

    const T* data() const noexcept {
        return ptr_;
    }

    std::size_t size() const noexcept {
        return count_;
    }

private:
    // cudaFree cannot report errors from a destructor, so cleanup is best effort.
    void release() noexcept {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

#endif

env = Environment()

env.Tool('cuda')
env.Append(LIBPATH=['/usr/lib/nvidia-current'])
env.Append(LIBS=['cuda', 'cudart'])
debug_flags = ['-g', '-O0']
env.Append(NVCCFLAGS=['-arch', 'sm_13', '-pg'] + debug_flags)
env.Append(CPPFLAGS=debug_flags)
env.Append(LINKFLAGS=debug_flags)

kernels = [env.Object('park-miller_device.cu')]

cutil = [
'/home/christian/local/opt/NVIDIA_GPU_Computing_SDK/C/common/src/bank_checker.cpp',
'/home/christian/local/opt/NVIDIA_GPU_Computing_SDK/C/common/src/cmd_arg_reader.cpp',
#'/home/christian/local/opt/NVIDIA_GPU_Computing_SDK/C/common/src/cuda_runtime_dynlink.cpp',
'/home/christian/local/opt/NVIDIA_GPU_Computing_SDK/C/common/src/cutil.cpp',
#'/home/christian/local/opt/NVIDIA_GPU_Computing_SDK/C/common/src/multithreading.cpp',
#'/home/christian/local/opt/NVIDIA_GPU_Computing_SDK/C/common/src/param.cpp',
#'/home/christian/local/opt/NVIDIA_GPU_Computing_SDK/C/common/src/paramgl.cpp',
#'/home/christian/local/opt/NVIDIA_GPU_Computing_SDK/C/common/src/rendercheck_d3d10.cpp',
#'/home/christian/local/opt/NVIDIA_GPU_Computing_SDK/C/common/src/rendercheck_d3d11.cpp',
#'/home/christian/local/opt/NVIDIA_GPU_Computing_SDK/C/common/src/rendercheck_d3d9.cpp',
#'/home/christian/local/opt/NVIDIA_GPU_Computing_SDK/C/common/src/rendercheck_gl.cpp',
'/home/christian/local/opt/NVIDIA_GPU_Computing_SDK/C/common/src/stopwatch.cpp',
'/home/christian/local/opt/NVIDIA_GPU_Computing_SDK/C/common/src/stopwatch_linux.cpp',
]


program = env.Program('park-miller', Glob('*.cpp') 
                            + cutil
                            + kernels)

ptx = env.Ptx('park-miller_device.cu')
elf = env.Elf('park-miller_device.cu')
cubin = env.Cubin(elf)
txt = env.DeCubin(cubin)

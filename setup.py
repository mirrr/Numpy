from distutils.core import setup, Extension
import sysconfig

def main():
	CFLAGS = ['-g', '-Wall', '-std=c99', '-fopenmp', '-mavx', '-mfma', '-pthread', '-O3']
	LDFLAGS = ['-fopenmp']
	# Use the setup function we imported and set up the modules.
	# You may find this reference helpful: https://docs.python.org/3.6/extending/building.html
	# TODO: YOUR CODE HERE
	module1 = Extension('numc',
                    # define_macros = [('MAJOR_VERSION', '1'),
                    #                  ('MINOR_VERSION', '0')],
                    # include_dirs = ['/usr/local/include'],
                    # library_dirs = ['/usr/local/lib'],
                    sources = ['matrix.c', 'numc.c'],
                    extra_compile_args = CFLAGS,
                    extra_link_args = LDFLAGS)

	setup (name = 'numc',
       version = '1.0',
       description = 'This is a numc package',
       author = 'Jenny Miao and Miranda Cheung',
       author_email = 'jenn3888@berkeley.edu',
       ext_modules = [module1])

if __name__ == "__main__":
    main()

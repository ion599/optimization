from distutils.core import setup, Extension
import numpy

# define the extension module
cos_module_np = Extension('cos_module_np', sources=['cos_module_np.c'],
                                  include_dirs=[numpy.get_include()])
simplex_projection = Extension("simplex_projection", sources = ["simplex_projection.c"],
                                  include_dirs=[numpy.get_include()])

setup(
   ext_modules = [
       simplex_projection,
       cos_module_np,
   ],
)

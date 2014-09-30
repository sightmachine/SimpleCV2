from setuptools import setup, find_packages

import simplecv


setup(name="simplecv",
      version=simplecv.__version__,
      download_url='https://github.com/sightmachine/SimpleCV/zipball/1.3',
      description="Make Computers See with simplecv, the Python Framework for Machine Vision",
      long_description=("Framework for computer (machine) vision in Python, providing a unified, pythonic interface "
                        "to image acquisition, conversion, manipulation, and feature extraction."),
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Manufacturing',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Topic :: Multimedia :: Graphics :: Capture :: Digital Camera',
          'Topic :: Multimedia :: Graphics :: Graphics Conversion',
          'Topic :: Scientific/Engineering :: Image Recognition',
          'Topic :: Software Development :: Libraries :: Python Modules'],
      keywords='opencv, cv, machine vision, computer vision, image recognition, kinect, freenect',
      author='Sight Machine Inc',
      author_email='support@sightmachine.com',
      url='http://simplecv.org',
      license='BSD',
      packages=find_packages(exclude=['ez_setup']),
      zip_safe=False,
      requires=[
          'cv2',
          'cv',
          'numpy',
          'scipy',
          'pygame',
          'pil',
          'svgwrite',
          'skimage'
      ],
      package_data={  # DO NOT REMOVE, NEEDED TO LOAD INLINE FILES i = Image('simplecv')
          'simplecv': [
              'data/fonts/astloch/*',
              'data/fonts/la_belle_aurore/*',
              'data/fonts/monofett/*',
              'data/fonts/carter_one/*',
              'data/fonts/unifrakturmaguntia/*',
              'data/fonts/reenie_beanie/*',
              'data/fonts/special_elite/*',
              'data/fonts/shadows_into_light/*',
              'data/fonts/wire_one/*',
              'data/fonts/vt323/*',
              'data/fonts/kranky/*',
              'data/fonts/ubuntu/*',
              'data/fonts/wallpoet/*',
              'data/test/StereoVision/*',
              'data/test/standard/*',
              'data/test/animation/*',
              'data/sampleimages/*',
          ],
      },
      entry_points={
          'console_scripts': [
              'simplecv = simplecv.shell.shell:main',
          ],
          'simplecv.image': [
              # built-in image modules
              'convert = simplecv.core.image.convert',
              'detection = simplecv.core.image.detection',
              'dft = simplecv.core.image.dft',
              'filter = simplecv.core.image.filter',
              'operation = simplecv.core.image.operation',
              'track = simplecv.core.image.track',
              'transform = simplecv.core.image.transform',

              # built-in plugins
              'denoise = simplecv.plugins.denoise',
              'stega = simplecv.plugins.stega',
              'uploader = simplecv.plugins.uploader',
          ],
          'simplecv.renderer.renderers': [
              'pygame = simplecv.core.drawing.renderer.pygamerenderer:PyGameRenderer',
              'svg = simplecv.core.drawing.renderer.svgrenderer:SvgRenderer',
          ]
      },
      )

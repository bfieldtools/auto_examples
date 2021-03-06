.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_thermal_noise_discs_validation.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_thermal_noise_discs_validation.py:


Disc validation
=========================

This example validates the thermal noise computations against the analytical solution for a thin disc.




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    array([8.18720179e-15, 6.15787344e-15, 4.79918738e-15, 3.84517023e-15,
           3.14973105e-15, 2.62721394e-15, 2.22470252e-15, 1.90807557e-15,
           1.65453005e-15, 1.44835760e-15, 1.27845158e-15, 1.13677928e-15,
           1.01741481e-15, 9.15909166e-16, 8.28869603e-16, 7.53672383e-16,
           6.88262718e-16, 6.31012896e-16, 5.80620023e-16, 5.36031236e-16,
           4.96388260e-16, 4.60985795e-16, 4.29239923e-16, 4.00663881e-16,
           3.74849288e-16, 3.51451470e-16, 3.30177901e-16, 3.10779008e-16,
           2.93040824e-16, 2.76779059e-16, 2.61834295e-16, 2.48068063e-16,
           2.35359628e-16, 2.23603326e-16, 2.12706366e-16, 2.02586990e-16,
           1.93172940e-16, 1.84400156e-16, 1.76211692e-16, 1.68556785e-16,
           1.61390067e-16, 1.54670891e-16, 1.48362754e-16, 1.42432802e-16,
           1.36851397e-16, 1.31591749e-16, 1.26629593e-16, 1.21942910e-16,
           1.17511681e-16, 1.13317672e-16])





|


.. code-block:: default


    import numpy as np


    kB = 1.38064852e-23  # Boltzman constant
    mu0 = 4 * np.pi * 1e-7
    d = 100e-9  # Film thickness in meters
    T = 273 + 160  # Temperature in C


    r = 100e-6

    z = np.linspace(0.2e-3, 1.7e-3)


    def plat_sigma(T):
        ref_T = 293  # K
        ref_rho = 1.06e-7  # ohm*meter
        alpha = 0.00392  # 1/K

        rho = alpha * (T - ref_T) * ref_rho + ref_rho

        return 1 / rho


    mu0 * np.sqrt((3 * plat_sigma(T) * kB * T * d) / (2048 * np.pi)) * (2 * r) / (z ** 2)


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.005 seconds)


.. _sphx_glr_download_auto_examples_thermal_noise_discs_validation.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: discs_validation.py <discs_validation.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: discs_validation.ipynb <discs_validation.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_

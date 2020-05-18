.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_line_conductor_inductance.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_line_conductor_inductance.py:


Mutual inductance of wire loops
===============================

In this example, we demonstrate how to compute the mutual inductance between two sets of polyline wire loops.


.. code-block:: default



    from bfieldtools.line_conductor import LineConductor
    from bfieldtools.mesh_conductor import StreamFunction
    from bfieldtools.suhtools import SuhBasis
    from bfieldtools.utils import load_example_mesh

    import numpy as np
    from mayavi import mlab








We create a set of wire loops by picking a single (arbitrary) surface-harmonic mode
from a plane mesh.  Finally, we discretize the  mode into a set of wire loops, which we plot.


.. code-block:: default


    mesh = load_example_mesh("10x10_plane")
    mesh.apply_scale(0.1)  # Downsize from 10 meters to 1 meter
    N_contours = 20

    sb = SuhBasis(mesh, 10)  # Construct surface-harmonics basis
    sf = StreamFunction(
        sb.basis[:, 1], sb.mesh_conductor
    )  # Turn single mode into a stream function
    c = LineConductor(
        mesh=mesh, scalars=sf, N_contours=N_contours
    )  # Discretize the stream function into wire loops


    # Plot loops for testing
    c.plot_loops(origin=np.array([0, -100, 0]))





.. image:: /auto_examples/images/sphx_glr_line_conductor_inductance_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Calculating surface harmonics expansion...
    Computing the laplacian matrix...
    Computing the mass matrix...

    <mayavi.core.scene.Scene object at 0x7fa208a59e90>



Now, we create a shifted copy of the wire loops, and the calculate the
mutual_inductance between two sets of line conductors


.. code-block:: default



    mesh2 = mesh.copy()
    mesh2.vertices[:, 1] += 1
    c2 = LineConductor(mesh=mesh2, scalars=sf, N_contours=N_contours)
    fig = c.plot_loops(origin=np.array([0, -100, 0]))
    c2.plot_loops(figure=fig, origin=np.array([0, -100, 0]))

    Mself = c.line_mutual_inductance(c, separate_loops=True, radius=1e-3)
    M2 = c.line_mutual_inductance(c2, separate_loops=True)




.. image:: /auto_examples/images/sphx_glr_line_conductor_inductance_002.png
    :class: sphx-glr-single-img





Now, we plot the inductance matrices


.. code-block:: default


    import matplotlib.pyplot as plt

    ff, ax = plt.subplots(1, 2, figsize=(12, 8))
    plt.sca(ax[0])
    plt.matshow(Mself, fignum=0)
    plt.title("Inductance matrix of the first set of wire loops")
    plt.sca(ax[1])
    plt.matshow(M2, fignum=0)
    plt.title("Mutual inductance matrix between the sets of wire loops")

    ff.tight_layout()




.. image:: /auto_examples/images/sphx_glr_line_conductor_inductance_003.png
    :class: sphx-glr-single-img





The inductance derived from the continous current density
---------------------------------------------------------
1) Magnetic energy of a inductor is E = 0.5*L*I^2
2) For unit current I=1 the inductance is L=2*E
3) The total current of a stream function (sf) integrated over
   the from minimum to maximum is dsf = max(sf) - min(sf)
4) When discretized to N conductors the current per conductor is
   I =  dsf / N
5) When sf is normalized such that I=1, i.e., dsf = N
   the inductance approximated by the continous stream function is
   L = 2*sf.magnetic_energy


.. code-block:: default


    scaling = N_contours / (sf.max() - sf.min())
    L_approx = 2 * sf.magnetic_energy * (scaling ** 2)

    print("Inductance based on the continuous current density", L_approx)
    print("Inductance based on r=1mm wire", np.sum(Mself))




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Computing the inductance matrix...
    Computing self-inductance matrix using rough quadrature (degree=2).              For higher accuracy, set quad_degree to 4 or more.
    Estimating 2432 MiB required for 676 by 676 vertices...
    Computing inductance matrix in 20 chunks (7537 MiB memory free),              when approx_far=True using more chunks is faster...
    Computing triangle-coupling matrix
    Inductance matrix computation took 0.90 seconds.
    Inductance based on the continuous current density 8.689344781849715e-05
    Inductance based on r=1mm wire 9.793656583088348e-05





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  5.811 seconds)

**Estimated memory usage:**  134 MB


.. _sphx_glr_download_auto_examples_line_conductor_inductance.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: line_conductor_inductance.py <line_conductor_inductance.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: line_conductor_inductance.ipynb <line_conductor_inductance.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_publication_software_pub_biplanar_coil_design.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_publication_software_pub_biplanar_coil_design.py:


Biplanar coil design
====================

First example in the paper, showing a basic biplanar coil producing homogeneous field in a target
region between the two coil planes.


.. code-block:: default

    PLOT = True
    SAVE_FIGURES = False

    SAVE_DIR = "./Biplanar coil/"


    import numpy as np
    from mayavi import mlab
    import trimesh

    from bfieldtools.mesh_conductor import MeshConductor
    from bfieldtools.coil_optimize import optimize_streamfunctions
    from bfieldtools.contour import scalar_contour
    from bfieldtools.viz import plot_3d_current_loops
    from bfieldtools.utils import load_example_mesh, combine_meshes


    # Set unit, e.g. meter or millimeter.
    # This doesn't matter, the problem is scale-invariant
    scaling_factor = 1


    # Load simple plane mesh that is centered on the origin
    planemesh = load_example_mesh("10x10_plane_hires")

    planemesh.apply_scale(scaling_factor * 1.6)

    # Specify coil plane geometry
    center_offset = np.array([0, 0, 0]) * scaling_factor
    standoff = np.array([0, 5, 0]) * scaling_factor

    # Create coil plane pairs
    coil_plus = trimesh.Trimesh(
        planemesh.vertices + center_offset + standoff, planemesh.faces, process=False
    )

    coil_minus = trimesh.Trimesh(
        planemesh.vertices + center_offset - standoff, planemesh.faces, process=False
    )

    joined_planes = combine_meshes((coil_plus, coil_minus))

    joined_planes = joined_planes.subdivide()

    # Create mesh class object
    coil = MeshConductor(
        verts=joined_planes.vertices,
        tris=joined_planes.faces,
        fix_normals=True,
        basis_name="suh",
        N_suh=100,
    )





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Calculating surface harmonics expansion...
    Computing the laplacian matrix...
    Computing the mass matrix...




Set up target and stray field points


.. code-block:: default


    # Here, the target points are on a volumetric grid within a sphere

    center = np.array([0, 0, 0]) * scaling_factor

    sidelength = 3 * scaling_factor
    n = 8
    xx = np.linspace(-sidelength / 2, sidelength / 2, n)
    yy = np.linspace(-sidelength / 2, sidelength / 2, n)
    zz = np.linspace(-sidelength / 2, sidelength / 2, n)
    X, Y, Z = np.meshgrid(xx, yy, zz, indexing="ij")

    x = X.ravel()
    y = Y.ravel()
    z = Z.ravel()

    target_points = np.array([x, y, z]).T

    # Turn cube into sphere by rejecting points "in the corners"
    target_points = (
        target_points[np.linalg.norm(target_points, axis=1) < sidelength / 2] + center
    )


    #    #Here, the stray field points are on a spherical surface
    stray_radius = 20 * scaling_factor

    stray_points_mesh = trimesh.creation.icosphere(subdivisions=3, radius=stray_radius)
    stray_points = stray_points_mesh.vertices + center

    n_stray_points = len(stray_points)









Plot geometry


.. code-block:: default

    if PLOT:
        f = mlab.figure(None, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5), size=(800, 800))

        coil.plot_mesh(representation="wireframe", opacity=0.1, color=(0, 0, 0), figure=f)
        coil.plot_mesh(representation="surface", opacity=0.1, color=(0, 0, 0), figure=f)
        mlab.points3d(*target_points.T, color=(0, 0, 1))
        mlab.points3d(*stray_points.T, scale_factor=0.3, color=(1, 0, 0))

        f.scene.isometric_view()
        f.scene.camera.zoom(1.5)

        if SAVE_FIGURES:
            mlab.savefig(
                SAVE_DIR + "biplanar_geometry.png", figure=f, magnification=4,
            )
            mlab.close()





.. image:: /auto_examples/publication_software/images/sphx_glr_pub_biplanar_coil_design_001.png
    :class: sphx-glr-single-img





Create bfield specifications used when optimizing the coil geometry


.. code-block:: default


    # The absolute target field amplitude is not of importance,
    # and it is scaled to match the C matrix in the optimization function

    target_field = np.zeros(target_points.shape)
    target_field[:, 0] += 1  # Homogeneous field on X-axis


    target_spec = {
        "coupling": coil.B_coupling(target_points),
        "abs_error": 0.005,
        "target": target_field,
    }
    stray_spec = {
        "coupling": coil.B_coupling(stray_points),
        "abs_error": 0.01,
        "target": np.zeros((n_stray_points, 3)),
    }

    bfield_specification = [target_spec, stray_spec]





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Computing magnetic field coupling matrix, 12442 vertices by 160 target points... took 1.04 seconds.
    Computing magnetic field coupling matrix, 12442 vertices by 642 target points... took 2.50 seconds.




Run QP solver


.. code-block:: default

    import mosek

    coil.s, prob = optimize_streamfunctions(
        coil,
        [target_spec, stray_spec],
        objective=(0, 1),
        solver="MOSEK",
        solver_opts={"mosek_params": {mosek.iparam.num_threads: 8}},
    )





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Computing the resistance matrix...
    Pre-existing problem not passed, creating...
    Passing parameters to problem...
    Passing problem to solver...


    Problem
      Name                   :                 
      Objective sense        : min             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 4914            
      Cones                  : 1               
      Scalar variables       : 203             
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer started.
    Problem
      Name                   :                 
      Objective sense        : min             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 4914            
      Cones                  : 1               
      Scalar variables       : 203             
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer  - threads                : 8               
    Optimizer  - solved problem         : the dual        
    Optimizer  - Constraints            : 101
    Optimizer  - Cones                  : 1
    Optimizer  - Scalar variables       : 4914              conic                  : 102             
    Optimizer  - Semi-definite variables: 0                 scalarized             : 0               
    Factor     - setup time             : 0.02              dense det. time        : 0.00            
    Factor     - ML order time          : 0.00              GP order time          : 0.00            
    Factor     - nonzeros before factor : 5151              after factor           : 5151            
    Factor     - dense dim.             : 0                 flops                  : 2.43e+07        
    ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME  
    0   4.1e+01  1.0e+00  2.0e+00  0.00e+00   0.000000000e+00   -1.000000000e+00  1.0e+00  0.10  
    1   3.0e+01  7.4e-01  1.6e+00  -7.57e-01  9.177045197e-01   2.023636783e-01   7.4e-01  0.10  
    2   2.2e+01  5.3e-01  1.2e+00  -5.73e-01  7.093710486e+00   6.645559845e+00   5.3e-01  0.11  
    3   1.7e+01  4.1e-01  9.1e-01  -3.88e-01  1.880728023e+01   1.852786243e+01   4.1e-01  0.12  
    4   1.3e+01  3.1e-01  6.8e-01  -2.27e-01  8.221059630e+01   8.206193922e+01   3.1e-01  0.12  
    5   7.8e+00  1.9e-01  4.0e-01  -7.11e-02  1.082335786e+02   1.082836485e+02   1.9e-01  0.13  
    6   3.8e+00  9.3e-02  1.6e-01  2.07e-01   3.187214859e+02   3.188465640e+02   9.3e-02  0.13  
    7   1.3e+00  3.2e-02  4.3e-02  5.17e-01   4.933418323e+02   4.934997919e+02   3.2e-02  0.15  
    8   9.2e-01  2.3e-02  3.6e-02  -1.26e-01  6.772964342e+02   6.776150977e+02   2.3e-02  0.15  
    9   2.6e-01  6.4e-03  1.0e-02  -1.24e-02  1.887569070e+03   1.888038492e+03   6.4e-03  0.16  
    10  2.2e-01  5.3e-03  8.3e-03  3.23e-01   2.140727395e+03   2.141202529e+03   5.3e-03  0.17  
    11  6.6e-02  1.6e-03  2.3e-03  2.75e-01   3.921000149e+03   3.921431567e+03   1.6e-03  0.18  
    12  3.2e-02  7.8e-04  1.1e-03  1.96e-01   5.106619456e+03   5.107060048e+03   7.8e-04  0.19  
    13  5.2e-03  1.3e-04  1.2e-04  3.03e-01   7.663842369e+03   7.664042048e+03   1.3e-04  0.20  
    14  2.4e-03  5.9e-05  3.8e-05  8.45e-01   8.213316704e+03   8.213418403e+03   5.9e-05  0.21  
    15  2.1e-03  5.2e-05  3.5e-05  8.36e-01   8.235249341e+03   8.235357889e+03   5.2e-05  0.21  
    16  6.4e-04  1.6e-05  5.9e-06  6.16e-01   8.658971306e+03   8.659005720e+03   1.6e-05  0.22  
    17  1.4e-05  3.4e-07  2.4e-08  8.63e-01   8.855997986e+03   8.855999131e+03   3.4e-07  0.23  
    18  1.0e-06  2.5e-08  5.0e-10  9.99e-01   8.860961720e+03   8.860961809e+03   2.5e-08  0.24  
    19  7.9e-07  1.3e-09  2.9e-11  1.00e+00   8.861345397e+03   8.861345401e+03   1.3e-09  0.26  
    20  1.5e-06  9.8e-10  2.0e-11  1.00e+00   8.861350672e+03   8.861350675e+03   9.8e-10  0.27  
    21  3.1e-06  4.9e-10  1.1e-11  1.00e+00   8.861358630e+03   8.861358631e+03   4.9e-10  0.29  
    22  3.1e-06  4.9e-10  9.3e-12  1.00e+00   8.861358693e+03   8.861358695e+03   4.9e-10  0.31  
    23  3.1e-06  4.9e-10  4.1e-12  1.00e+00   8.861358697e+03   8.861358699e+03   4.9e-10  0.34  
    24  3.2e-06  4.9e-10  1.2e-12  1.00e+00   8.861358728e+03   8.861358730e+03   4.9e-10  0.36  
    25  3.2e-06  4.9e-10  1.2e-12  1.00e+00   8.861358728e+03   8.861358730e+03   4.9e-10  0.39  
    26  3.3e-06  4.6e-10  2.1e-12  1.00e+00   8.861359223e+03   8.861359225e+03   4.6e-10  0.41  
    27  3.8e-06  4.0e-10  6.3e-12  1.00e+00   8.861360152e+03   8.861360153e+03   4.0e-10  0.43  
    28  3.9e-06  4.0e-10  7.3e-12  1.00e+00   8.861360165e+03   8.861360166e+03   4.0e-10  0.45  
    29  3.9e-06  4.0e-10  7.3e-12  1.00e+00   8.861360165e+03   8.861360166e+03   4.0e-10  0.48  
    30  3.9e-06  4.0e-10  7.3e-12  1.00e+00   8.861360165e+03   8.861360166e+03   4.0e-10  0.51  
    31  3.9e-06  3.9e-10  6.0e-12  1.00e+00   8.861360266e+03   8.861360268e+03   3.9e-10  0.53  
    32  3.9e-06  3.9e-10  6.0e-12  1.00e+00   8.861360266e+03   8.861360268e+03   3.9e-10  0.55  
    33  3.9e-06  3.9e-10  6.0e-12  1.00e+00   8.861360266e+03   8.861360268e+03   3.9e-10  0.58  
    Optimizer terminated. Time: 0.61    


    Interior-point solution summary
      Problem status  : PRIMAL_AND_DUAL_FEASIBLE
      Solution status : OPTIMAL
      Primal.  obj: 8.8613602663e+03    nrm: 2e+04    Viol.  con: 2e-08    var: 0e+00    cones: 0e+00  
      Dual.    obj: 8.8613602680e+03    nrm: 5e+05    Viol.  con: 0e+00    var: 3e-08    cones: 0e+00  




Plot coil windings and target points


.. code-block:: default



    N_contours = 6

    loops = scalar_contour(coil.mesh, coil.s.vert, N_contours=N_contours)

    if PLOT:
        f = mlab.figure(None, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5), size=(650, 750))
        mlab.clf()

        plot_3d_current_loops(loops, colors="auto", figure=f)

        # B_target = coil.B_coupling(target_points) @ coil.s

        # mlab.quiver3d(*target_points.T, *B_target.T, mode="arrow", scale_factor=1)

        f.scene.isometric_view()
        #    f.scene.camera.zoom(0.95)
        if SAVE_FIGURES:
            mlab.savefig(
                SAVE_DIR + "biplanar_loops.png", figure=f, magnification=4,
            )

            mlab.close()





.. image:: /auto_examples/publication_software/images/sphx_glr_pub_biplanar_coil_design_002.png
    :class: sphx-glr-single-img





Plot continuous stream function


.. code-block:: default


    if PLOT:
        from bfieldtools.viz import plot_data_on_vertices

        f = mlab.figure(None, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5), size=(800, 800))
        mlab.clf()

        plot_data_on_vertices(coil.mesh, coil.s.vert, figure=f, ncolors=256)

        f.scene.camera.parallel_projection = 1
        mlab.view(90, 90)
        f.scene.camera.zoom(1.5)

        if SAVE_FIGURES:
            mlab.savefig(
                SAVE_DIR + "biplanar_streamfunction.png", figure=f, magnification=4,
            )

            mlab.close()



.. image:: /auto_examples/publication_software/images/sphx_glr_pub_biplanar_coil_design_003.png
    :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  19.106 seconds)

**Estimated memory usage:**  1300 MB


.. _sphx_glr_download_auto_examples_publication_software_pub_biplanar_coil_design.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: pub_biplanar_coil_design.py <pub_biplanar_coil_design.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: pub_biplanar_coil_design.ipynb <pub_biplanar_coil_design.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_

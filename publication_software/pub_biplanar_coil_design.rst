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

    Computing magnetic field coupling matrix, 12442 vertices by 160 target points... took 1.08 seconds.
    Computing magnetic field coupling matrix, 12442 vertices by 642 target points... took 2.52 seconds.




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
    /home/rzetter/miniconda3/lib/python3.7/site-packages/cvxpy-1.1.0a3-py3.7-linux-x86_64.egg/cvxpy/reductions/solvers/solving_chain.py:170: UserWarning: You are solving a parameterized problem that is not DPP. Because the problem is not DPP, subsequent solves will not be faster than the first one.
      "You are solving a parameterized problem that is not DPP. "


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
    Factor     - setup time             : 0.05              dense det. time        : 0.00            
    Factor     - ML order time          : 0.00              GP order time          : 0.00            
    Factor     - nonzeros before factor : 5151              after factor           : 5151            
    Factor     - dense dim.             : 0                 flops                  : 2.47e+07        
    ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME  
    0   1.0e+01  1.0e+00  2.0e+00  0.00e+00   0.000000000e+00   -1.000000000e+00  1.0e+00  0.56  
    1   5.1e+00  5.0e-01  8.0e-01  -2.34e-01  2.497994602e+00   1.933623472e+00   5.0e-01  0.71  
    2   2.6e+00  2.5e-01  3.0e-01  4.25e-01   2.191892141e+01   2.159001252e+01   2.5e-01  0.72  
    3   1.3e+00  1.3e-01  1.1e-01  8.11e-01   3.579359296e+01   3.561938993e+01   1.3e-01  0.73  
    4   6.9e-01  6.8e-02  4.3e-02  9.62e-01   5.455194307e+01   5.446788094e+01   6.8e-02  0.74  
    5   2.4e-01  2.4e-02  1.2e-02  9.72e-01   7.409541473e+01   7.407281308e+01   2.4e-02  0.75  
    6   1.4e-01  1.3e-02  6.9e-03  -1.24e-01  1.121043462e+02   1.121026249e+02   1.3e-02  0.76  
    7   3.9e-02  3.8e-03  2.1e-03  -2.23e-01  2.554515935e+02   2.554886317e+02   3.8e-03  0.77  
    8   2.8e-02  2.8e-03  1.6e-03  1.74e-01   3.017059230e+02   3.017502652e+02   2.8e-03  0.77  
    9   2.4e-02  2.3e-03  1.3e-03  9.52e-02   3.334017556e+02   3.334495649e+02   2.3e-03  0.78  
    10  8.9e-03  8.7e-04  4.4e-04  1.51e-01   5.381771635e+02   5.382236338e+02   8.7e-04  0.79  
    11  2.8e-03  2.7e-04  1.3e-04  2.56e-02   7.922159023e+02   7.922591402e+02   2.7e-04  0.80  
    12  5.5e-04  5.4e-05  1.6e-05  4.15e-01   1.057080129e+03   1.057100774e+03   5.4e-05  0.81  
    13  3.5e-04  3.4e-05  8.6e-06  8.56e-01   1.098547860e+03   1.098561812e+03   3.4e-05  0.82  
    14  2.8e-04  2.7e-05  6.7e-06  8.46e-01   1.108211450e+03   1.108225099e+03   2.7e-05  0.82  
    15  1.3e-04  1.3e-05  2.2e-06  5.65e-01   1.150499370e+03   1.150506178e+03   1.3e-05  0.83  
    16  1.1e-05  1.0e-06  5.5e-08  7.96e-01   1.188783129e+03   1.188783764e+03   1.0e-06  0.84  
    17  6.5e-09  3.9e-09  7.7e-13  9.82e-01   1.192709273e+03   1.192709273e+03   6.4e-10  0.85  
    18  3.3e-09  1.9e-09  1.9e-13  1.00e+00   1.192710493e+03   1.192710494e+03   3.2e-10  0.87  
    19  8.6e-09  1.2e-11  1.3e-14  1.00e+00   1.192711718e+03   1.192711718e+03   1.4e-13  0.89  
    Optimizer terminated. Time: 0.90    


    Interior-point solution summary
      Problem status  : PRIMAL_AND_DUAL_FEASIBLE
      Solution status : OPTIMAL
      Primal.  obj: 1.1927117176e+03    nrm: 2e+03    Viol.  con: 4e-12    var: 0e+00    cones: 0e+00  
      Dual.    obj: 1.1927117176e+03    nrm: 7e+04    Viol.  con: 0e+00    var: 4e-10    cones: 0e+00  




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

   **Total running time of the script:** ( 0 minutes  19.026 seconds)


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

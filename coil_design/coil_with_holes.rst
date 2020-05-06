.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_coil_design_coil_with_holes.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_coil_design_coil_with_holes.py:


Coil with interior holes
========================

Example showing a basic biplanar coil producing homogeneous field in a target
region between the two coil planes. The coil planes have holes in them,


.. code-block:: default


    import numpy as np
    import trimesh

    from bfieldtools.mesh_conductor import MeshConductor
    from bfieldtools.coil_optimize import optimize_streamfunctions
    from bfieldtools.utils import combine_meshes, load_example_mesh


    # Load simple plane mesh that is centered on the origin
    planemesh = load_example_mesh("plane_w_holes")

    angle = np.pi / 2
    rotation_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )

    planemesh.apply_transform(rotation_matrix)

    # Specify coil plane geometry
    center_offset = np.array([0, 0, 0])
    standoff = np.array([0, 20, 0])

    # Create coil plane pairs
    coil_plus = trimesh.Trimesh(
        planemesh.vertices + center_offset + standoff, planemesh.faces, process=False
    )

    coil_minus = trimesh.Trimesh(
        planemesh.vertices + center_offset - standoff, planemesh.faces, process=False
    )

    joined_planes = combine_meshes((coil_plus, coil_minus))

    # Create MeshConductor object, which finds the holes and sets the boundary condition
    coil = MeshConductor(mesh_obj=joined_planes, fix_normals=True)








Set up target and stray field points


.. code-block:: default


    # Here, the target points are on a volumetric grid within a sphere

    center = np.array([0, 0, 0])

    sidelength = 10
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









Create bfield specifications used when optimizing the coil geometry


.. code-block:: default


    # The absolute target field amplitude is not of importance,
    # and it is scaled to match the C matrix in the optimization function

    target_field = np.zeros(target_points.shape)
    target_field[:, 0] = target_field[:, 0] + 1

    target_abs_error = np.zeros_like(target_field)
    target_abs_error[:, 0] += 0.001
    target_abs_error[:, 1:3] += 0.005

    target_spec = {
        "coupling": coil.B_coupling(target_points),
        "abs_error": target_abs_error,
        "target": target_field,
    }

    bfield_specification = [target_spec]





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Computing magnetic field coupling matrix, 2772 vertices by 160 target points... took 0.21 seconds.




Run QP solver


.. code-block:: default

    import mosek

    coil.s, prob = optimize_streamfunctions(
        coil,
        bfield_specification,
        objective="minimum_inductive_energy",
        solver="MOSEK",
        solver_opts={"mosek_params": {mosek.iparam.num_threads: 8}},
    )





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Computing the inductance matrix...
    Computing self-inductance matrix using rough quadrature (degree=2).              For higher accuracy, set quad_degree to 4 or more.
    Estimating 27549 MiB required for 2772 by 2772 vertices...
    Computing inductance matrix in 60 chunks (12026 MiB memory free),              when approx_far=True using more chunks is faster...
    Computing triangle-coupling matrix
    Inductance matrix computation took 9.81 seconds.
    Pre-existing problem not passed, creating...
    Passing parameters to problem...
    Passing problem to solver...
    /home/rzetter/miniconda3/lib/python3.7/site-packages/cvxpy-1.1.0a3-py3.7-linux-x86_64.egg/cvxpy/reductions/solvers/solving_chain.py:170: UserWarning: You are solving a parameterized problem that is not DPP. Because the problem is not DPP, subsequent solves will not be faster than the first one.
      "You are solving a parameterized problem that is not DPP. "


    Problem
      Name                   :                 
      Objective sense        : min             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 3364            
      Cones                  : 1               
      Scalar variables       : 4807            
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer started.
    Problem
      Name                   :                 
      Objective sense        : min             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 3364            
      Cones                  : 1               
      Scalar variables       : 4807            
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer  - threads                : 8               
    Optimizer  - solved problem         : the dual        
    Optimizer  - Constraints            : 2403
    Optimizer  - Cones                  : 1
    Optimizer  - Scalar variables       : 3364              conic                  : 2404            
    Optimizer  - Semi-definite variables: 0                 scalarized             : 0               
    Factor     - setup time             : 0.44              dense det. time        : 0.00            
    Factor     - ML order time          : 0.12              GP order time          : 0.00            
    Factor     - nonzeros before factor : 2.89e+06          after factor           : 2.89e+06        
    Factor     - dense dim.             : 0                 flops                  : 2.13e+10        
    ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME  
    0   3.2e+01  1.0e+00  2.0e+00  0.00e+00   0.000000000e+00   -1.000000000e+00  1.0e+00  63.16 
    1   2.2e+01  6.8e-01  1.0e+00  -4.93e-02  4.394350998e+00   3.612244501e+00   6.8e-01  63.50 
    2   1.4e+01  4.5e-01  1.2e-01  1.73e-01   1.742209642e+01   1.685672440e+01   4.5e-01  63.84 
    3   4.1e+00  1.3e-01  5.6e-02  1.35e+00   2.924843157e+01   2.903652904e+01   1.3e-01  64.18 
    4   1.5e+00  4.6e-02  1.6e-02  5.05e-01   5.617431064e+01   5.608964506e+01   4.6e-02  64.54 
    5   2.3e-01  7.3e-03  1.2e-03  3.38e-01   9.172543086e+01   9.170633095e+01   7.3e-03  64.95 
    6   8.0e-02  2.5e-03  3.4e-04  7.43e-01   1.029694889e+02   1.029645986e+02   2.5e-03  65.28 
    7   2.6e-03  8.0e-05  2.1e-06  1.02e+00   1.085962470e+02   1.085961045e+02   8.0e-05  65.74 
    8   1.4e-03  4.2e-05  8.3e-07  1.01e+00   1.086891419e+02   1.086890775e+02   4.2e-05  66.07 
    9   2.0e-04  6.2e-06  5.4e-08  1.01e+00   1.087870964e+02   1.087870918e+02   6.2e-06  66.45 
    10  2.7e-05  8.6e-07  2.9e-09  1.00e+00   1.088016474e+02   1.088016470e+02   8.6e-07  66.84 
    11  3.9e-06  1.9e-07  1.6e-10  1.00e+00   1.088036432e+02   1.088036431e+02   1.2e-07  67.22 
    12  9.0e-07  7.3e-08  1.6e-11  1.00e+00   1.088038961e+02   1.088038960e+02   2.8e-08  67.55 
    13  2.3e-07  1.8e-08  2.0e-12  1.00e+00   1.088039536e+02   1.088039536e+02   7.1e-09  68.01 
    14  6.3e-08  2.1e-09  2.1e-13  1.00e+00   1.088039708e+02   1.088039708e+02   7.8e-10  68.46 
    Optimizer terminated. Time: 68.62   


    Interior-point solution summary
      Problem status  : PRIMAL_AND_DUAL_FEASIBLE
      Solution status : OPTIMAL
      Primal.  obj: 1.0880397081e+02    nrm: 2e+02    Viol.  con: 1e-09    var: 0e+00    cones: 0e+00  
      Dual.    obj: 1.0880397082e+02    nrm: 3e+03    Viol.  con: 0e+00    var: 8e-10    cones: 0e+00  




Plot the computed streamfunction


.. code-block:: default


    coil.s.plot(ncolors=256)



.. image:: /auto_examples/coil_design/images/sphx_glr_coil_with_holes_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <mayavi.modules.surface.Surface object at 0x7f969cd803b0>




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 1 minutes  31.411 seconds)


.. _sphx_glr_download_auto_examples_coil_design_coil_with_holes.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: coil_with_holes.py <coil_with_holes.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: coil_with_holes.ipynb <coil_with_holes.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_

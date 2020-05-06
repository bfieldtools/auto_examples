.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_coil_design_mamba_coil_design.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_coil_design_mamba_coil_design.py:


MAMBA coil
==========

Compact example of a biplanar coil producing homogeneous field in a number of target
regions arranged in a grid. Meant to demonstrate the flexibility in target choice, inspired by the 
technique "multiple-acquisition micro B(0) array" (MAMBA) technique, see https://doi.org/10.1002/mrm.10464


.. code-block:: default



    import numpy as np
    from mayavi import mlab
    import trimesh


    from bfieldtools.mesh_conductor import MeshConductor
    from bfieldtools.coil_optimize import optimize_streamfunctions
    from bfieldtools.contour import scalar_contour
    from bfieldtools.viz import plot_3d_current_loops

    from bfieldtools.utils import combine_meshes, load_example_mesh


    # Load simple plane mesh that is centered on the origin
    planemesh = load_example_mesh("10x10_plane_hires")

    # Specify coil plane geometry
    center_offset = np.array([0, 0, 0])
    standoff = np.array([0, 1.5, 0])

    # Create coil plane pairs
    coil_plus = trimesh.Trimesh(
        planemesh.vertices + center_offset + standoff, planemesh.faces, process=False
    )

    coil_minus = trimesh.Trimesh(
        planemesh.vertices + center_offset - standoff, planemesh.faces, process=False
    )

    joined_planes = combine_meshes((coil_plus, coil_minus))

    # Create mesh class object
    coil = MeshConductor(mesh_obj=joined_planes, fix_normals=True, basis_name="inner")








Set up target and stray field points. Here, the target points are on a planar
4x4 grid slightly smaller than the coil dimensions.


.. code-block:: default


    center = np.array([0, 0, 0])

    sidelength = 0.5
    n = 4

    height = 0.1
    n_height = 2
    xx = np.linspace(-sidelength / 2, sidelength / 2, n)
    yy = np.linspace(-height / 2, height / 2, n_height)
    zz = np.linspace(-sidelength / 2, sidelength / 2, n)
    X, Y, Z = np.meshgrid(xx, yy, zz, indexing="ij")

    x = X.ravel()
    y = Y.ravel()
    z = Z.ravel()

    target_points = np.array([x, y, z]).T


    grid_target_points = list()
    target_field = list()

    hori_offsets = [-3, -1, 1, 3]
    vert_offsets = [-3, -1, 1, 3]

    for i, offset_x in enumerate(hori_offsets):
        for j, offset_y in enumerate(vert_offsets):
            grid_target_points.append(target_points + np.array([offset_x, 0, offset_y]))
            target_field.append((i + j - 3) * np.ones((len(target_points),)))

    target_points = np.asarray(grid_target_points).reshape((-1, 3))
    target_field = np.asarray(target_field).reshape((-1,))

    target_field = np.array(
        [np.zeros((len(target_field),)), target_field, np.zeros((len(target_field),))]
    ).T


    target_abs_error = np.zeros_like(target_field)
    target_abs_error[:, 1] += 0.1
    target_abs_error[:, 0::2] += 0.1








Plot target points and mesh


.. code-block:: default

    coil.plot_mesh(opacity=0.1)
    mlab.quiver3d(*target_points.T, *target_field.T)





.. image:: /auto_examples/coil_design/images/sphx_glr_mamba_coil_design_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <mayavi.modules.vectors.Vectors object at 0x7f969e732650>



Compute coupling matrix that is used to compute the generated magnetic field, create field specification


.. code-block:: default



    target_spec = {
        "coupling": coil.B_coupling(target_points),
        "abs_error": target_abs_error,
        "target": target_field,
    }





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Computing magnetic field coupling matrix, 3184 vertices by 512 target points... took 0.52 seconds.




Run QP solver, plot result


.. code-block:: default


    import mosek

    coil.s, prob = optimize_streamfunctions(
        coil,
        [target_spec],
        objective="minimum_inductive_energy",
        solver="MOSEK",
        solver_opts={"mosek_params": {mosek.iparam.num_threads: 8}},
    )


    coil.s.plot()

    coil.s.discretize(N_contours=10).plot_loops()



.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/coil_design/images/sphx_glr_mamba_coil_design_002.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/coil_design/images/sphx_glr_mamba_coil_design_003.png
            :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Computing the inductance matrix...
    Computing self-inductance matrix using rough quadrature (degree=2).              For higher accuracy, set quad_degree to 4 or more.
    Estimating 34964 MiB required for 3184 by 3184 vertices...
    Computing inductance matrix in 60 chunks (12047 MiB memory free),              when approx_far=True using more chunks is faster...
    Computing triangle-coupling matrix
    Inductance matrix computation took 13.20 seconds.
    Pre-existing problem not passed, creating...
    Passing parameters to problem...
    Passing problem to solver...
    /home/rzetter/miniconda3/lib/python3.7/site-packages/cvxpy-1.1.0a3-py3.7-linux-x86_64.egg/cvxpy/reductions/solvers/solving_chain.py:170: UserWarning: You are solving a parameterized problem that is not DPP. Because the problem is not DPP, subsequent solves will not be faster than the first one.
      "You are solving a parameterized problem that is not DPP. "


    Problem
      Name                   :                 
      Objective sense        : min             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 5970            
      Cones                  : 1               
      Scalar variables       : 5795            
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer started.
    Problem
      Name                   :                 
      Objective sense        : min             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 5970            
      Cones                  : 1               
      Scalar variables       : 5795            
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer  - threads                : 8               
    Optimizer  - solved problem         : the dual        
    Optimizer  - Constraints            : 2897
    Optimizer  - Cones                  : 1
    Optimizer  - Scalar variables       : 5970              conic                  : 2898            
    Optimizer  - Semi-definite variables: 0                 scalarized             : 0               
    Factor     - setup time             : 0.98              dense det. time        : 0.00            
    Factor     - ML order time          : 0.17              GP order time          : 0.00            
    Factor     - nonzeros before factor : 4.20e+06          after factor           : 4.20e+06        
    Factor     - dense dim.             : 0                 flops                  : 4.53e+10        
    ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME  
    0   2.5e+01  1.0e+00  2.0e+00  0.00e+00   0.000000000e+00   -1.000000000e+00  1.0e+00  112.73
    1   5.9e+00  2.4e-01  2.8e-01  1.12e-01   1.190431429e+02   1.187399022e+02   2.4e-01  113.41
    2   5.9e-01  2.4e-02  8.3e-03  7.37e-01   1.581771497e+02   1.581383257e+02   2.4e-02  114.04
    3   8.4e-02  3.4e-03  4.5e-04  9.94e-01   1.512418314e+02   1.512362841e+02   3.4e-03  114.67
    4   1.9e-02  7.6e-04  4.9e-05  1.00e+00   1.498224865e+02   1.498212790e+02   7.6e-04  115.29
    5   1.5e-03  5.9e-05  1.3e-06  9.99e-01   1.493643784e+02   1.493643279e+02   5.9e-05  116.03
    6   4.9e-04  2.0e-05  2.5e-07  1.00e+00   1.493706983e+02   1.493706821e+02   2.0e-05  116.67
    7   6.5e-05  2.6e-06  1.2e-08  1.00e+00   1.493735721e+02   1.493735700e+02   2.6e-06  117.38
    8   4.5e-06  1.8e-07  2.3e-10  1.00e+00   1.493739910e+02   1.493739909e+02   1.8e-07  118.15
    9   6.3e-07  2.5e-08  7.8e-12  1.00e+00   1.493740222e+02   1.493740222e+02   2.5e-08  118.82
    10  3.1e-07  1.3e-08  1.0e-13  1.00e+00   1.493740255e+02   1.493740255e+02   1.3e-08  119.76
    11  1.5e-07  1.9e-09  3.8e-13  1.00e+00   1.493740287e+02   1.493740285e+02   3.6e-12  120.36
    Optimizer terminated. Time: 120.74  


    Interior-point solution summary
      Problem status  : PRIMAL_AND_DUAL_FEASIBLE
      Solution status : OPTIMAL
      Primal.  obj: 1.4937402870e+02    nrm: 3e+02    Viol.  con: 2e-11    var: 0e+00    cones: 0e+00  
      Dual.    obj: 1.4937402850e+02    nrm: 7e+01    Viol.  con: 2e-11    var: 9e-10    cones: 0e+00  

    <mayavi.core.scene.Scene object at 0x7f969ee6d350>




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 2 minutes  41.128 seconds)


.. _sphx_glr_download_auto_examples_coil_design_mamba_coil_design.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: mamba_coil_design.py <mamba_coil_design.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: mamba_coil_design.ipynb <mamba_coil_design.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_

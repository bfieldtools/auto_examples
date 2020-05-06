.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_coil_design_spherical_harmonics_coil_design.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_coil_design_spherical_harmonics_coil_design.py:


Spherical harmonics-generating coil design
==========================================

Example showing a basic biplanar coil producing a field profile defined by
spherical harmonics. We use the surface harmonics basis for the stream function,
and optimize the coupling to spherical harmonics components, thus creating a compact
optimization problem that can be solved very quickly.


.. code-block:: default


    import numpy as np
    import trimesh

    from bfieldtools.mesh_conductor import MeshConductor
    from bfieldtools.coil_optimize import optimize_streamfunctions
    from bfieldtools.utils import combine_meshes, load_example_mesh


    # Load simple plane mesh that is centered on the origin
    planemesh = load_example_mesh("10x10_plane_hires")

    # Specify coil plane geometry
    center_offset = np.array([0, 0, 0])
    standoff = np.array([0, 3, 0])

    # Create coil plane pairs
    coil_plus = trimesh.Trimesh(
        planemesh.vertices + center_offset + standoff, planemesh.faces, process=False
    )

    coil_minus = trimesh.Trimesh(
        planemesh.vertices + center_offset - standoff, planemesh.faces, process=False
    )

    joined_planes = combine_meshes((coil_plus, coil_minus))


    # To spice things up, let's distort the planes a bit
    joined_planes.vertices = (
        joined_planes.vertices
        - 0.5
        * np.linalg.norm(joined_planes.vertices, axis=1)[:, None]
        * joined_planes.vertex_normals
    )

    # Create mesh class object
    coil = MeshConductor(
        mesh_obj=joined_planes, fix_normals=True, basis_name="suh", N_suh=100
    )






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Calculating surface harmonics expansion...
    Computing the laplacian matrix...
    Computing the mass matrix...




Set up target spherical harmonics components


.. code-block:: default


    target_alms = np.zeros((coil.opts["N_sph"] * (coil.opts["N_sph"] + 2),))
    target_blms = np.zeros((coil.opts["N_sph"] * (coil.opts["N_sph"] + 2),))

    target_blms[3] += 1









Create bfield specifications used when optimizing the coil geometry


.. code-block:: default



    target_spec = {
        "coupling": coil.sph_couplings[1],
        "abs_error": 0.01,
        "target": target_blms,
    }






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Computing coupling matrices
    l = 1 computed
    l = 2 computed
    l = 3 computed
    l = 4 computed
    l = 5 computed




Run QP solver


.. code-block:: default

    import mosek

    coil.s, prob = optimize_streamfunctions(
        coil,
        [target_spec],
        objective="minimum_ohmic_power",
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
      Constraints            : 172             
      Cones                  : 1               
      Scalar variables       : 203             
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer started.
    Problem
      Name                   :                 
      Objective sense        : min             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 172             
      Cones                  : 1               
      Scalar variables       : 203             
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer  - threads                : 8               
    Optimizer  - solved problem         : the dual        
    Optimizer  - Constraints            : 101
    Optimizer  - Cones                  : 1
    Optimizer  - Scalar variables       : 172               conic                  : 102             
    Optimizer  - Semi-definite variables: 0                 scalarized             : 0               
    Factor     - setup time             : 0.00              dense det. time        : 0.00            
    Factor     - ML order time          : 0.00              GP order time          : 0.00            
    Factor     - nonzeros before factor : 5151              after factor           : 5151            
    Factor     - dense dim.             : 0                 flops                  : 1.14e+06        
    ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME  
    0   4.0e+00  1.0e+00  2.0e+00  0.00e+00   0.000000000e+00   -1.000000000e+00  1.0e+00  0.02  
    1   1.6e+00  3.8e-01  6.8e-01  1.38e-01   1.666552819e+00   1.243296373e+00   3.8e-01  0.02  
    2   7.3e-01  1.8e-01  2.5e-01  2.67e-01   6.563789939e+00   6.363463675e+00   1.8e-01  0.03  
    3   1.9e-01  4.7e-02  3.1e-02  2.08e-01   1.598167160e+01   1.587692435e+01   4.7e-02  0.03  
    4   7.0e-03  1.7e-03  9.1e-05  1.33e+00   1.728266852e+01   1.727695742e+01   1.7e-03  0.03  
    5   2.3e-04  5.7e-05  7.7e-07  1.14e+00   1.736734555e+01   1.736719791e+01   5.7e-05  0.03  
    6   3.0e-06  7.5e-07  1.1e-09  1.00e+00   1.737061779e+01   1.737061575e+01   7.5e-07  0.03  
    7   1.4e-07  3.5e-08  1.1e-11  1.00e+00   1.737066005e+01   1.737065996e+01   3.5e-08  0.03  
    8   2.5e-09  1.4e-10  4.8e-14  1.00e+00   1.737066229e+01   1.737066230e+01   1.4e-10  0.03  
    Optimizer terminated. Time: 0.03    


    Interior-point solution summary
      Problem status  : PRIMAL_AND_DUAL_FEASIBLE
      Solution status : OPTIMAL
      Primal.  obj: 1.7370662293e+01    nrm: 4e+01    Viol.  con: 2e-10    var: 0e+00    cones: 0e+00  
      Dual.    obj: 1.7370662298e+01    nrm: 2e+02    Viol.  con: 3e-08    var: 2e-10    cones: 0e+00  




Plot coil windings


.. code-block:: default


    coil.s.discretize(N_contours=8).plot_loops()



.. image:: /auto_examples/coil_design/images/sphx_glr_spherical_harmonics_coil_design_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <mayavi.core.scene.Scene object at 0x7f969e013dd0>




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  22.114 seconds)


.. _sphx_glr_download_auto_examples_coil_design_spherical_harmonics_coil_design.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: spherical_harmonics_coil_design.py <spherical_harmonics_coil_design.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: spherical_harmonics_coil_design.ipynb <spherical_harmonics_coil_design.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_

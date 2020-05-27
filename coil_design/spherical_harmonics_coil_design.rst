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
    Optimizer  - solved problem         : the primal      
    Optimizer  - Constraints            : 36
    Optimizer  - Cones                  : 1
    Optimizer  - Scalar variables       : 137               conic                  : 102             
    Optimizer  - Semi-definite variables: 0                 scalarized             : 0               
    Factor     - setup time             : 0.00              dense det. time        : 0.00            
    Factor     - ML order time          : 0.00              GP order time          : 0.00            
    Factor     - nonzeros before factor : 666               after factor           : 666             
    Factor     - dense dim.             : 0                 flops                  : 1.44e+05        
    ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME  
    0   8.9e+00  1.0e+00  1.0e+00  0.00e+00   5.000000000e-01   5.000000000e-01   1.0e+00  0.02  
    1   3.7e+00  4.2e-01  4.3e-01  -3.09e-01  1.326877957e+00   1.692364855e+00   4.2e-01  0.03  
    2   2.4e+00  2.6e-01  2.6e-01  -1.48e-01  4.896405523e+00   5.345147693e+00   2.6e-01  0.03  
    3   1.1e+00  1.2e-01  1.1e-01  -8.59e-02  1.325296342e+01   1.374685656e+01   1.2e-01  0.03  
    4   8.0e-01  9.0e-02  7.3e-02  5.10e-01   1.772841716e+01   1.811172365e+01   9.0e-02  0.03  
    5   1.1e-01  1.2e-02  2.9e-03  9.63e-01   2.556973711e+01   2.558921374e+01   1.2e-02  0.03  
    6   8.0e-03  8.9e-04  5.2e-05  1.09e+00   2.693424318e+01   2.693516231e+01   8.9e-04  0.03  
    7   4.7e-05  5.2e-06  2.1e-08  1.08e+00   2.704665381e+01   2.704665557e+01   5.2e-06  0.03  
    8   1.6e-06  1.8e-07  1.3e-10  1.00e+00   2.704724320e+01   2.704724325e+01   1.8e-07  0.03  
    9   4.6e-08  5.1e-09  6.6e-13  1.00e+00   2.704726335e+01   2.704726335e+01   5.2e-09  0.03  
    10  1.3e-11  6.9e-10  1.2e-16  1.00e+00   2.704726401e+01   2.704726401e+01   8.6e-13  0.03  
    Optimizer terminated. Time: 0.03    


    Interior-point solution summary
      Problem status  : PRIMAL_AND_DUAL_FEASIBLE
      Solution status : OPTIMAL
      Primal.  obj: 2.7047264009e+01    nrm: 6e+01    Viol.  con: 2e-11    var: 0e+00    cones: 0e+00  
      Dual.    obj: 2.7047264009e+01    nrm: 3e+02    Viol.  con: 6e-10    var: 2e-09    cones: 0e+00  




Plot coil windings


.. code-block:: default


    coil.s.discretize(N_contours=8).plot_loops()



.. image:: /auto_examples/coil_design/images/sphx_glr_spherical_harmonics_coil_design_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <mayavi.core.scene.Scene object at 0x7f0c11db0d10>




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  23.156 seconds)

**Estimated memory usage:**  67 MB


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

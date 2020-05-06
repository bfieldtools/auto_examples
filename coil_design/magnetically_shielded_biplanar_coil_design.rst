.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_coil_design_magnetically_shielded_biplanar_coil_design.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_coil_design_magnetically_shielded_biplanar_coil_design.py:


Magnetically shielded  coil
===========================
Compact example of design of a biplanar coil within a cylindrical shield.
The effect of the shield is prospectively taken into account while designing the coil.
The coil is positioned close to the end of the shield to demonstrate the effect


.. code-block:: default



    import numpy as np
    from mayavi import mlab
    import trimesh


    from bfieldtools.mesh_conductor import MeshConductor, StreamFunction
    from bfieldtools.coil_optimize import optimize_streamfunctions
    from bfieldtools.contour import scalar_contour
    from bfieldtools.viz import plot_3d_current_loops, plot_data_on_vertices
    from bfieldtools.utils import combine_meshes

    import pkg_resources


    # Set unit, e.g. meter or millimeter.
    # This doesn't matter, the problem is scale-invariant
    scaling_factor = 1


    # Load simple plane mesh that is centered on the origin
    planemesh = trimesh.load(
        file_obj=pkg_resources.resource_filename(
            "bfieldtools", "example_meshes/10x10_plane_hires.obj"
        ),
        process=False,
    )

    planemesh.apply_scale(scaling_factor)

    # Specify coil plane geometry
    center_offset = np.array([9, 0, 0]) * scaling_factor
    standoff = np.array([0, 4, 0]) * scaling_factor

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

    # Separate object for shield geometry
    shieldmesh = trimesh.load(
        file_obj=pkg_resources.resource_filename(
            "bfieldtools", "example_meshes/closed_cylinder_remeshed.stl"
        ),
        process=True,
    )
    shieldmesh.apply_scale(15)

    shield = MeshConductor(
        mesh_obj=shieldmesh, process=True, fix_normals=True, basis_name="vertex"
    )









Set up target  points and plot geometry


.. code-block:: default


    # Here, the target points are on a volumetric grid within a sphere
    # Set up target and stray field points

    # Here, the target points are on a volumetric grid within a sphere

    center = np.array([9, 0, 0]) * scaling_factor

    sidelength = 3 * scaling_factor
    n = 12
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


    # Plot coil, shield and target points

    f = mlab.figure(None, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5), size=(800, 800))

    coil.plot_mesh(representation="surface")
    shield.plot_mesh(representation="surface", cull_front=True, color=(0.9, 0.9, 0.9))
    mlab.points3d(*target_points.T)


    f.scene.isometric_view()
    f.scene.camera.zoom(1.2)





.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/coil_design/images/sphx_glr_magnetically_shielded_biplanar_coil_design_001.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/coil_design/images/sphx_glr_magnetically_shielded_biplanar_coil_design_002.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/coil_design/images/sphx_glr_magnetically_shielded_biplanar_coil_design_003.png
            :class: sphx-glr-multi-img





Let's design a coil without taking the magnetic shield into account


.. code-block:: default


    # The absolute target field amplitude is not of importance,
    # and it is scaled to match the C matrix in the optimization function
    target_field = np.zeros(target_points.shape)
    target_field[:, 0] = target_field[:, 0] + 1  # Homogeneous Y-field


    target_abs_error = np.zeros_like(target_field)
    target_abs_error[:, 0] += 0.005
    target_abs_error[:, 1:3] += 0.01

    target_spec = {
        "coupling": coil.B_coupling(target_points),
        "rel_error": 0,
        "abs_error": target_abs_error,
        "target": target_field,
    }

    import mosek

    coil.s, coil.prob = optimize_streamfunctions(
        coil,
        [target_spec],
        objective="minimum_inductive_energy",
        solver="MOSEK",
        solver_opts={"mosek_params": {mosek.iparam.num_threads: 8}},
    )






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Computing magnetic field coupling matrix, 3184 vertices by 672 target points... took 0.63 seconds.
    Computing the inductance matrix...
    Computing self-inductance matrix using rough quadrature (degree=2).              For higher accuracy, set quad_degree to 4 or more.
    Estimating 34964 MiB required for 3184 by 3184 vertices...
    Computing inductance matrix in 60 chunks (12884 MiB memory free),              when approx_far=True using more chunks is faster...
    Computing triangle-coupling matrix
    Inductance matrix computation took 13.64 seconds.
    Pre-existing problem not passed, creating...
    Passing parameters to problem...
    Passing problem to solver...
    /home/rzetter/miniconda3/lib/python3.7/site-packages/cvxpy-1.1.0a3-py3.7-linux-x86_64.egg/cvxpy/reductions/solvers/solving_chain.py:170: UserWarning: You are solving a parameterized problem that is not DPP. Because the problem is not DPP, subsequent solves will not be faster than the first one.
      "You are solving a parameterized problem that is not DPP. "


    Problem
      Name                   :                 
      Objective sense        : min             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 6930            
      Cones                  : 1               
      Scalar variables       : 5795            
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer started.
    Problem
      Name                   :                 
      Objective sense        : min             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 6930            
      Cones                  : 1               
      Scalar variables       : 5795            
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer  - threads                : 8               
    Optimizer  - solved problem         : the dual        
    Optimizer  - Constraints            : 2897
    Optimizer  - Cones                  : 1
    Optimizer  - Scalar variables       : 6930              conic                  : 2898            
    Optimizer  - Semi-definite variables: 0                 scalarized             : 0               
    Factor     - setup time             : 1.23              dense det. time        : 0.00            
    Factor     - ML order time          : 0.19              GP order time          : 0.00            
    Factor     - nonzeros before factor : 4.20e+06          after factor           : 4.20e+06        
    Factor     - dense dim.             : 0                 flops                  : 4.93e+10        
    ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME  
    0   1.3e+02  1.0e+00  2.0e+00  0.00e+00   0.000000000e+00   -1.000000000e+00  1.0e+00  114.96
    1   6.2e+01  4.9e-01  8.3e-01  -1.96e-01  1.240457322e+02   1.235079730e+02   4.9e-01  115.80
    2   2.8e+01  2.2e-01  2.8e-01  -8.66e-02  4.745038929e+02   4.742160042e+02   2.2e-01  116.54
    3   6.9e+00  5.3e-02  3.7e-02  1.16e+00   7.772866775e+02   7.772419801e+02   5.3e-02  117.41
    4   1.3e+00  1.0e-02  3.0e-03  1.00e+00   8.795694177e+02   8.795594215e+02   1.0e-02  118.28
    5   1.8e-01  1.4e-03  1.6e-04  9.78e-01   9.028714779e+02   9.028704012e+02   1.4e-03  119.00
    6   1.9e-02  1.5e-04  5.4e-06  9.99e-01   9.066099589e+02   9.066098492e+02   1.5e-04  119.88
    7   2.4e-03  1.9e-05  2.5e-07  1.00e+00   9.070392461e+02   9.070392326e+02   1.9e-05  120.58
    8   1.3e-04  9.9e-07  2.8e-09  1.00e+00   9.070997059e+02   9.070997053e+02   9.9e-07  121.28
    9   2.1e-06  3.6e-09  5.6e-11  1.00e+00   9.071030696e+02   9.071030686e+02   3.6e-09  122.44
    Optimizer terminated. Time: 122.92  


    Interior-point solution summary
      Problem status  : PRIMAL_AND_DUAL_FEASIBLE
      Solution status : OPTIMAL
      Primal.  obj: 9.0710306957e+02    nrm: 2e+03    Viol.  con: 6e-09    var: 0e+00    cones: 0e+00  
      Dual.    obj: 9.0710306864e+02    nrm: 6e+03    Viol.  con: 1e-07    var: 1e-09    cones: 0e+00  




Plot coil windings and target points


.. code-block:: default


    loops = scalar_contour(coil.mesh, coil.s.vert, N_contours=10)

    f = mlab.figure(None, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5), size=(800, 800))
    mlab.clf()

    plot_3d_current_loops(loops, colors="auto", figure=f)

    B_target = coil.B_coupling(target_points) @ coil.s

    mlab.quiver3d(*target_points.T, *B_target.T, mode="arrow", scale_factor=0.75)


    f.scene.isometric_view()
    f.scene.camera.zoom(0.95)




.. image:: /auto_examples/coil_design/images/sphx_glr_magnetically_shielded_biplanar_coil_design_004.png
    :class: sphx-glr-single-img





Now, let's compute the effect of the shield on the field produced by the coil


.. code-block:: default


    # Points slightly inside the shield
    d = (
        np.mean(np.diff(shield.mesh.vertices[shield.mesh.faces[:, 0:2]], axis=1), axis=0)
        / 10
    )
    points = shield.mesh.vertices - d * shield.mesh.vertex_normals


    # Solve equivalent stream function for the perfect linear mu-metal layer.
    # This is the equivalent surface current in the shield that would cause its
    # scalar magnetic potential to be constant
    shield.s = StreamFunction(
        np.linalg.solve(shield.U_coupling(points), coil.U_coupling(points) @ coil.s), shield
    )





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Computing scalar potential coupling matrix, 2773 vertices by 2773 target points... took 9.21 seconds.
    Computing scalar potential coupling matrix, 3184 vertices by 2773 target points... took 10.08 seconds.




Plot the difference in field when taking the shield into account


.. code-block:: default


    f = mlab.figure(None, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5), size=(800, 800))
    mlab.clf()

    B_target = coil.B_coupling(target_points) @ coil.s

    B_target_w_shield = (
        coil.B_coupling(target_points) @ coil.s
        + shield.B_coupling(target_points) @ shield.s
    )

    B_quiver = mlab.quiver3d(
        *target_points.T,
        *(B_target_w_shield - B_target).T,
        colormap="viridis",
        mode="arrow"
    )
    f.scene.isometric_view()
    mlab.colorbar(B_quiver, title="Difference in magnetic field (a.u.)")




.. image:: /auto_examples/coil_design/images/sphx_glr_magnetically_shielded_biplanar_coil_design_005.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Computing magnetic field coupling matrix, 2773 vertices by 672 target points... took 0.58 seconds.
    This object has no scalar data

    <mayavi.core.lut_manager.LUTManager object at 0x7f969e109230>



Let's redesign the coil taking the shield into account prospectively


.. code-block:: default


    shield.coupling = np.linalg.solve(shield.U_coupling(points), coil.U_coupling(points))

    secondary_C = shield.B_coupling(target_points) @ shield.coupling

    total_C = coil.B_coupling(target_points) + secondary_C

    target_spec_w_shield = {
        "coupling": total_C,
        "rel_error": 0,
        "abs_error": target_abs_error,
        "target": target_field,
    }


    coil.s2, coil.prob2 = optimize_streamfunctions(
        coil,
        [target_spec_w_shield],
        objective="minimum_inductive_energy",
        solver="MOSEK",
        solver_opts={"mosek_params": {mosek.iparam.num_threads: 8}},
    )





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Pre-existing problem not passed, creating...
    Passing parameters to problem...
    Passing problem to solver...


    Problem
      Name                   :                 
      Objective sense        : min             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 6930            
      Cones                  : 1               
      Scalar variables       : 5795            
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer started.
    Problem
      Name                   :                 
      Objective sense        : min             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 6930            
      Cones                  : 1               
      Scalar variables       : 5795            
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer  - threads                : 8               
    Optimizer  - solved problem         : the dual        
    Optimizer  - Constraints            : 2897
    Optimizer  - Cones                  : 1
    Optimizer  - Scalar variables       : 6930              conic                  : 2898            
    Optimizer  - Semi-definite variables: 0                 scalarized             : 0               
    Factor     - setup time             : 1.19              dense det. time        : 0.00            
    Factor     - ML order time          : 0.16              GP order time          : 0.00            
    Factor     - nonzeros before factor : 4.20e+06          after factor           : 4.20e+06        
    Factor     - dense dim.             : 0                 flops                  : 4.93e+10        
    ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME  
    0   1.3e+02  1.0e+00  2.0e+00  0.00e+00   0.000000000e+00   -1.000000000e+00  1.0e+00  120.90
    1   6.6e+01  5.1e-01  9.2e-01  -2.59e-01  1.079606567e+02   1.074140904e+02   5.1e-01  121.62
    2   3.2e+01  2.5e-01  3.5e-01  -1.91e-01  4.304279912e+02   4.301298179e+02   2.5e-01  122.44
    3   2.0e+01  1.6e-01  1.8e-01  1.11e+00   6.401620263e+02   6.399871669e+02   1.6e-01  123.21
    4   1.2e+01  9.4e-02  8.6e-02  8.89e-01   8.113033668e+02   8.112046537e+02   9.4e-02  123.89
    5   9.4e+00  7.3e-02  6.0e-02  9.20e-01   8.666550972e+02   8.665809853e+02   7.3e-02  124.61
    6   5.8e+00  4.5e-02  3.2e-02  8.79e-01   9.379560115e+02   9.379205554e+02   4.5e-02  125.27
    7   7.8e-01  6.1e-03  1.7e-03  8.90e-01   1.095853913e+03   1.095850700e+03   6.1e-03  125.98
    8   1.3e-01  1.0e-03  1.2e-04  9.81e-01   1.122783264e+03   1.122782965e+03   1.0e-03  126.77
    9   1.8e-02  1.4e-04  6.0e-06  9.96e-01   1.128483616e+03   1.128483567e+03   1.4e-04  127.51
    10  2.8e-03  2.2e-05  3.9e-07  9.99e-01   1.129273255e+03   1.129273250e+03   2.2e-05  128.25
    11  5.9e-05  4.6e-07  1.0e-09  1.00e+00   1.129423574e+03   1.129423574e+03   4.6e-07  129.03
    12  5.0e-05  3.9e-07  6.1e-10  1.00e+00   1.129424027e+03   1.129424028e+03   3.9e-07  129.95
    13  1.8e-05  1.4e-07  3.9e-10  1.00e+00   1.129425791e+03   1.129425790e+03   1.4e-07  130.61
    14  1.0e-05  8.0e-08  1.9e-10  1.00e+00   1.129426188e+03   1.129426188e+03   8.0e-08  131.57
    15  9.1e-06  2.8e-08  1.9e-11  1.00e+00   1.129426711e+03   1.129426711e+03   5.0e-09  132.22
    16  8.3e-06  2.5e-08  3.1e-11  1.00e+00   1.129426715e+03   1.129426715e+03   4.4e-09  133.26
    17  6.7e-06  2.0e-08  5.8e-11  1.00e+00   1.129426723e+03   1.129426722e+03   3.3e-09  134.19
    18  4.2e-06  1.7e-08  6.5e-11  1.00e+00   1.129426723e+03   1.129426724e+03   3.3e-09  135.22
    19  4.2e-06  1.7e-08  6.5e-11  1.00e+00   1.129426723e+03   1.129426724e+03   3.3e-09  136.28
    20  5.4e-06  1.6e-08  3.4e-11  1.00e+00   1.129426725e+03   1.129426725e+03   3.1e-09  137.24
    21  7.3e-05  8.3e-09  1.9e-11  1.00e+00   1.129426735e+03   1.129426736e+03   1.5e-09  138.34
    Optimizer terminated. Time: 138.79  


    Interior-point solution summary
      Problem status  : PRIMAL_AND_DUAL_FEASIBLE
      Solution status : OPTIMAL
      Primal.  obj: 1.1294267352e+03    nrm: 2e+03    Viol.  con: 3e-09    var: 0e+00    cones: 0e+00  
      Dual.    obj: 1.1294267357e+03    nrm: 1e+04    Viol.  con: 3e-08    var: 4e-09    cones: 0e+00  




Plot the newly designed coil windings and field at the target points


.. code-block:: default


    loops = scalar_contour(coil.mesh, coil.s2.vert, N_contours=10)

    f = mlab.figure(None, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5), size=(800, 800))
    mlab.clf()

    plot_3d_current_loops(loops, colors="auto", figure=f)

    B_target2 = total_C @ coil.s2
    mlab.quiver3d(*target_points.T, *B_target2.T, mode="arrow", scale_factor=0.75)


    f.scene.isometric_view()
    f.scene.camera.zoom(0.95)





.. image:: /auto_examples/coil_design/images/sphx_glr_magnetically_shielded_biplanar_coil_design_006.png
    :class: sphx-glr-single-img





Plot difference in field


.. code-block:: default



    import seaborn as sns
    import matplotlib.pyplot as plt


    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    axnames = ["X", "Y", "Z"]

    # fig.suptitle('Component-wise effect of magnetic shield on target field amplitude distribution')
    for ax_idx, ax in enumerate(axes):

        sns.kdeplot(
            B_target[:, ax_idx],
            label="Coil without shield",
            ax=ax,
            shade=True,
            legend=False,
        )
        sns.kdeplot(
            B_target_w_shield[:, ax_idx],
            label="Coil with shield",
            ax=ax,
            shade=True,
            legend=False,
        )
        sns.kdeplot(
            B_target2[:, ax_idx],
            label="Coil designed with shield",
            ax=ax,
            shade=True,
            legend=False,
        )
        #    ax.set_title(axnames[ax_idx])
        ax.get_yaxis().set_visible(False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_xlabel("Magnetic field on %s-axis" % axnames[ax_idx])

        if ax_idx == 0:
            ax.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])



.. image:: /auto_examples/coil_design/images/sphx_glr_magnetically_shielded_biplanar_coil_design_007.png
    :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 6 minutes  4.111 seconds)


.. _sphx_glr_download_auto_examples_coil_design_magnetically_shielded_biplanar_coil_design.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: magnetically_shielded_biplanar_coil_design.py <magnetically_shielded_biplanar_coil_design.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: magnetically_shielded_biplanar_coil_design.ipynb <magnetically_shielded_biplanar_coil_design.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_

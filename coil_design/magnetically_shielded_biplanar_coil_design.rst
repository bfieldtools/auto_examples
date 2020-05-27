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

    coil.plot_mesh(representation="surface", figure=f, opacity=0.5)
    shield.plot_mesh(representation="surface", opacity=0.2, figure=f)
    mlab.points3d(*target_points.T)

    f.scene.isometric_view()
    f.scene.camera.zoom(1.1)





.. image:: /auto_examples/coil_design/images/sphx_glr_magnetically_shielded_biplanar_coil_design_001.png
    :class: sphx-glr-single-img





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
    Computing inductance matrix in 60 chunks (12151 MiB memory free),              when approx_far=True using more chunks is faster...
    Computing triangle-coupling matrix
    Inductance matrix computation took 13.03 seconds.
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
    Factor     - setup time             : 0.92              dense det. time        : 0.00            
    Factor     - ML order time          : 0.18              GP order time          : 0.00            
    Factor     - nonzeros before factor : 4.20e+06          after factor           : 4.20e+06        
    Factor     - dense dim.             : 0                 flops                  : 3.31e+10        
    ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME  
    0   1.0e+03  1.0e+00  2.0e+00  0.00e+00   0.000000000e+00   -1.000000000e+00  1.0e+00  4.42  
    1   4.5e+02  4.4e-01  1.1e+00  -7.30e-01  5.327102614e+01   5.318155897e+01   4.4e-01  5.04  
    2   2.3e+02  2.2e-01  6.0e-01  -4.50e-01  2.594248081e+02   2.599322310e+02   2.2e-01  5.63  
    3   3.1e+01  3.1e-02  6.6e-02  -8.59e-02  8.803728379e+02   8.812167455e+02   3.1e-02  6.21  
    4   2.7e+00  2.6e-03  1.7e-03  7.54e-01   9.125606569e+02   9.126395285e+02   2.6e-03  6.93  
    5   6.0e-01  5.8e-04  1.8e-04  9.79e-01   9.069071853e+02   9.069250230e+02   5.8e-04  7.55  
    6   2.7e-01  2.6e-04  5.5e-05  9.95e-01   9.072761546e+02   9.072838260e+02   2.6e-04  8.13  
    7   2.8e-02  2.7e-05  1.8e-06  9.98e-01   9.070886715e+02   9.070894398e+02   2.7e-05  8.74  
    8   2.8e-03  2.7e-06  5.6e-08  1.00e+00   9.070853236e+02   9.070853942e+02   2.7e-06  9.37  
    9   1.1e-04  1.1e-07  1.1e-09  1.00e+00   9.071023136e+02   9.071023144e+02   1.1e-07  9.97  
    10  6.1e-07  2.4e-09  2.1e-12  1.00e+00   9.071030780e+02   9.071030781e+02   5.9e-10  10.69 
    Optimizer terminated. Time: 11.02   


    Interior-point solution summary
      Problem status  : PRIMAL_AND_DUAL_FEASIBLE
      Solution status : OPTIMAL
      Primal.  obj: 9.0710307796e+02    nrm: 2e+03    Viol.  con: 4e-09    var: 0e+00    cones: 0e+00  
      Dual.    obj: 9.0710307806e+02    nrm: 6e+03    Viol.  con: 2e-06    var: 3e-08    cones: 0e+00  




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




.. image:: /auto_examples/coil_design/images/sphx_glr_magnetically_shielded_biplanar_coil_design_002.png
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

    Computing scalar potential coupling matrix, 2773 vertices by 2773 target points... took 9.00 seconds.
    Computing scalar potential coupling matrix, 3184 vertices by 2773 target points... took 9.78 seconds.




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




.. image:: /auto_examples/coil_design/images/sphx_glr_magnetically_shielded_biplanar_coil_design_003.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Computing magnetic field coupling matrix, 2773 vertices by 672 target points... took 0.56 seconds.
    This object has no scalar data

    <mayavi.core.lut_manager.LUTManager object at 0x7ff939c57e30>



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
    Factor     - setup time             : 0.90              dense det. time        : 0.00            
    Factor     - ML order time          : 0.17              GP order time          : 0.00            
    Factor     - nonzeros before factor : 4.20e+06          after factor           : 4.20e+06        
    Factor     - dense dim.             : 0                 flops                  : 3.31e+10        
    ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME  
    0   1.0e+03  1.0e+00  2.0e+00  0.00e+00   0.000000000e+00   -1.000000000e+00  1.0e+00  4.37  
    1   4.6e+02  4.4e-01  1.1e+00  -7.17e-01  6.091562708e+01   6.078906288e+01   4.4e-01  4.96  
    2   2.4e+02  2.4e-01  6.3e-01  -4.44e-01  2.794436170e+02   2.798668032e+02   2.4e-01  5.60  
    3   3.5e+01  3.4e-02  7.3e-02  -8.97e-02  1.021741496e+03   1.022519676e+03   3.4e-02  6.22  
    4   5.8e-01  5.6e-04  1.8e-04  7.38e-01   1.138954022e+03   1.138971891e+03   5.6e-04  6.96  
    5   1.6e-01  1.6e-04  2.3e-05  9.98e-01   1.129521654e+03   1.129524935e+03   1.6e-04  7.65  
    6   6.7e-02  6.5e-05  6.1e-06  9.99e-01   1.129644522e+03   1.129645924e+03   6.5e-05  8.26  
    7   8.8e-03  8.5e-06  2.9e-07  1.00e+00   1.129374023e+03   1.129374209e+03   8.5e-06  8.97  
    8   1.1e-03  1.1e-06  1.3e-08  1.00e+00   1.129416676e+03   1.129416701e+03   1.1e-06  9.60  
    9   3.4e-05  3.3e-08  8.2e-11  1.00e+00   1.129426431e+03   1.129426431e+03   3.3e-08  10.33 
    10  1.1e-05  1.1e-08  1.0e-10  1.00e+00   1.129426640e+03   1.129426639e+03   1.1e-08  10.93 
    11  1.7e-06  1.3e-09  1.2e-10  1.00e+00   1.129426733e+03   1.129426730e+03   1.3e-09  11.56 
    Optimizer terminated. Time: 11.89   


    Interior-point solution summary
      Problem status  : PRIMAL_AND_DUAL_FEASIBLE
      Solution status : OPTIMAL
      Primal.  obj: 1.1294267334e+03    nrm: 2e+03    Viol.  con: 8e-09    var: 0e+00    cones: 0e+00  
      Dual.    obj: 1.1294267301e+03    nrm: 1e+04    Viol.  con: 5e-06    var: 1e-08    cones: 0e+00  




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





.. image:: /auto_examples/coil_design/images/sphx_glr_magnetically_shielded_biplanar_coil_design_004.png
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



.. image:: /auto_examples/coil_design/images/sphx_glr_magnetically_shielded_biplanar_coil_design_005.png
    :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 1 minutes  30.986 seconds)

**Estimated memory usage:**  3778 MB


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

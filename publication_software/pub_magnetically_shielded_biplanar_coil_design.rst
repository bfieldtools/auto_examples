.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_publication_software_pub_magnetically_shielded_biplanar_coil_design.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_publication_software_pub_magnetically_shielded_biplanar_coil_design.py:


Magnetically shielded  coil
===========================
Compact example of design of a biplanar coil within a cylindrical shield.
The effect of the shield is prospectively taken into account while designing the coil.
The coil is positioned close to the end of the shield to demonstrate the effect


.. code-block:: default


    PLOT = True
    SAVE_FIGURES = False
    SAVE_STREAMFUNCTION = False

    SAVE_DIR = "./Shielded coil/"

    import numpy as np
    from mayavi import mlab
    import trimesh


    from bfieldtools.mesh_conductor import MeshConductor, StreamFunction
    from bfieldtools.coil_optimize import optimize_streamfunctions
    from bfieldtools.utils import load_example_mesh, combine_meshes

    from bfieldtools.contour import scalar_contour
    from bfieldtools.viz import plot_3d_current_loops

    import pkg_resources


    # Set unit, e.g. meter or millimeter.
    # This doesn't matter, the problem is scale-invariant
    scaling_factor = 1


    # Load simple plane mesh that is centered on the origin
    planemesh = load_example_mesh("10x10_plane_hires")
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

    joined_planes = combine_meshes((coil_minus, coil_plus))


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
    if PLOT:
        f = mlab.figure(None, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5), size=(800, 800))

        coil.plot_mesh(representation="surface", figure=f, opacity=0.15)
        shield.plot_mesh(
            representation="surface", cull_front=True, color=(0.9, 0.9, 0.9), figure=f
        )
        mlab.points3d(*target_points.T)

        f.scene.isometric_view()
        f.scene.camera.zoom(1.2)

        if SAVE_FIGURES:
            mlab.savefig(
                SAVE_DIR + "shielded_biplanar_geometry.png", figure=f, magnification=4,
            )
            mlab.close()





.. image:: /auto_examples/publication_software/images/sphx_glr_pub_magnetically_shielded_biplanar_coil_design_001.png
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


    if SAVE_STREAMFUNCTION:
        np.save(SAVE_DIR + "biplanar_streamfunction.npy", coil.s.vert)




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Computing magnetic field coupling matrix, 3184 vertices by 672 target points... took 0.65 seconds.
    Computing the inductance matrix...
    Computing self-inductance matrix using rough quadrature (degree=2).              For higher accuracy, set quad_degree to 4 or more.
    Estimating 34964 MiB required for 3184 by 3184 vertices...
    Computing inductance matrix in 60 chunks (12017 MiB memory free),              when approx_far=True using more chunks is faster...
    Computing triangle-coupling matrix
    Inductance matrix computation took 13.47 seconds.
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
    Factor     - setup time             : 1.35              dense det. time        : 0.00            
    Factor     - ML order time          : 0.18              GP order time          : 0.00            
    Factor     - nonzeros before factor : 4.20e+06          after factor           : 4.20e+06        
    Factor     - dense dim.             : 0                 flops                  : 4.93e+10        
    ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME  
    0   1.0e+03  1.0e+00  2.0e+00  0.00e+00   0.000000000e+00   -1.000000000e+00  1.0e+00  117.46
    1   4.5e+02  4.4e-01  1.1e+00  -7.30e-01  5.327108191e+01   5.318161480e+01   4.4e-01  118.22
    2   2.3e+02  2.2e-01  6.0e-01  -4.50e-01  2.594250451e+02   2.599324681e+02   2.2e-01  118.89
    3   3.1e+01  3.1e-02  6.6e-02  -8.59e-02  8.803737426e+02   8.812176503e+02   3.1e-02  119.54
    4   2.7e+00  2.6e-03  1.7e-03  7.54e-01   9.125617181e+02   9.126405893e+02   2.6e-03  120.33
    5   6.0e-01  5.8e-04  1.8e-04  9.79e-01   9.069082291e+02   9.069260669e+02   5.8e-04  121.01
    6   2.7e-01  2.6e-04  5.5e-05  9.95e-01   9.072772186e+02   9.072848896e+02   2.6e-04  121.69
    7   2.7e-02  2.7e-05  1.8e-06  9.98e-01   9.070897206e+02   9.070904889e+02   2.7e-05  122.37
    8   2.8e-03  2.7e-06  5.6e-08  1.00e+00   9.070863768e+02   9.070864475e+02   2.7e-06  123.06
    9   1.1e-04  1.1e-07  1.0e-11  1.00e+00   9.071033596e+02   9.071033638e+02   1.1e-07  123.90
    10  6.1e-07  6.0e-10  1.8e-12  1.00e+00   9.071041244e+02   9.071041243e+02   6.0e-10  124.92
    Optimizer terminated. Time: 125.36  


    Interior-point solution summary
      Problem status  : PRIMAL_AND_DUAL_FEASIBLE
      Solution status : OPTIMAL
      Primal.  obj: 9.0710412437e+02    nrm: 2e+03    Viol.  con: 4e-09    var: 0e+00    cones: 0e+00  
      Dual.    obj: 9.0710412432e+02    nrm: 6e+03    Viol.  con: 2e-06    var: 1e-09    cones: 0e+00  




Plot coil windings and target points


.. code-block:: default


    loops = scalar_contour(coil.mesh, coil.s.vert, N_contours=10)

    if PLOT:
        f = mlab.figure(None, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5), size=(800, 800))
        mlab.clf()

        plot_3d_current_loops(loops, colors="auto", figure=f)

        B_target = coil.B_coupling(target_points) @ coil.s

        mlab.quiver3d(*target_points.T, *B_target.T, mode="arrow", scale_factor=0.75)

        f.scene.isometric_view()
        f.scene.camera.zoom(0.95)

        if SAVE_FIGURES:
            mlab.savefig(
                SAVE_DIR + "shielded_biplanar_ignored.png", figure=f, magnification=4,
            )
            mlab.close()



.. image:: /auto_examples/publication_software/images/sphx_glr_pub_magnetically_shielded_biplanar_coil_design_002.png
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

    Computing scalar potential coupling matrix, 2773 vertices by 2773 target points... took 8.84 seconds.
    Computing scalar potential coupling matrix, 3184 vertices by 2773 target points... took 9.74 seconds.




Plot the difference in field when taking the shield into account


.. code-block:: default


    if PLOT:
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




.. image:: /auto_examples/publication_software/images/sphx_glr_pub_magnetically_shielded_biplanar_coil_design_003.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Computing magnetic field coupling matrix, 2773 vertices by 672 target points... took 0.54 seconds.
    This object has no scalar data




Let's redesign the coil taking the shield into account prospectively


.. code-block:: default


    shield.coupling = np.linalg.solve(shield.U_coupling(points), coil.U_coupling(points))

    secondary_C = shield.B_coupling(target_points) @ shield.coupling

    total_C = coil.B_coupling(target_points) + secondary_C

    target_spec_w_shield = {
        "coupling": total_C,
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
    Factor     - setup time             : 1.08              dense det. time        : 0.00            
    Factor     - ML order time          : 0.17              GP order time          : 0.00            
    Factor     - nonzeros before factor : 4.20e+06          after factor           : 4.20e+06        
    Factor     - dense dim.             : 0                 flops                  : 4.93e+10        
    ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME  
    0   1.0e+03  1.0e+00  2.0e+00  0.00e+00   0.000000000e+00   -1.000000000e+00  1.0e+00  112.63
    1   4.6e+02  4.4e-01  1.1e+00  -7.17e-01  6.091568992e+01   6.078912581e+01   4.4e-01  113.33
    2   2.4e+02  2.4e-01  6.3e-01  -4.44e-01  2.794438665e+02   2.798670529e+02   2.4e-01  114.00
    3   3.5e+01  3.4e-02  7.3e-02  -8.97e-02  1.021742465e+03   1.022520645e+03   3.4e-02  114.67
    4   5.8e-01  5.6e-04  1.8e-04  7.38e-01   1.138954809e+03   1.138972680e+03   5.6e-04  115.50
    5   1.6e-01  1.6e-04  2.3e-05  9.98e-01   1.129523039e+03   1.129526323e+03   1.6e-04  116.29
    6   6.7e-02  6.5e-05  6.1e-06  9.99e-01   1.129645660e+03   1.129647063e+03   6.5e-05  116.96
    7   8.8e-03  8.5e-06  2.9e-07  1.00e+00   1.129375305e+03   1.129375491e+03   8.5e-06  117.80
    8   1.1e-03  1.1e-06  1.3e-08  1.00e+00   1.129417794e+03   1.129417819e+03   1.1e-06  118.66
    9   3.5e-05  3.4e-08  7.7e-11  1.00e+00   1.129427523e+03   1.129427524e+03   3.4e-08  119.52
    10  1.1e-05  1.1e-08  2.3e-10  1.00e+00   1.129427736e+03   1.129427739e+03   1.1e-08  120.20
    11  1.3e-06  1.3e-09  5.8e-11  1.00e+00   1.129427831e+03   1.129427832e+03   1.3e-09  120.89
    Optimizer terminated. Time: 121.34  


    Interior-point solution summary
      Problem status  : PRIMAL_AND_DUAL_FEASIBLE
      Solution status : OPTIMAL
      Primal.  obj: 1.1294278308e+03    nrm: 2e+03    Viol.  con: 8e-09    var: 0e+00    cones: 0e+00  
      Dual.    obj: 1.1294278324e+03    nrm: 1e+04    Viol.  con: 5e-06    var: 1e-08    cones: 0e+00  




Plot the newly designed coil windings and field at the target points


.. code-block:: default


    loops = scalar_contour(coil.mesh, coil.s2.vert, N_contours=10)

    if PLOT:
        f = mlab.figure(None, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5), size=(800, 800))
        mlab.clf()

        plot_3d_current_loops(loops, colors="auto", figure=f)

        B_target2 = total_C @ coil.s2
        mlab.quiver3d(*target_points.T, *B_target2.T, mode="arrow", scale_factor=0.75)

        f.scene.isometric_view()
        f.scene.camera.zoom(0.95)
        if SAVE_FIGURES:
            mlab.savefig(
                SAVE_DIR + "shielded_biplanar_prospective.png", figure=f, magnification=4,
            )
            mlab.close()




.. image:: /auto_examples/publication_software/images/sphx_glr_pub_magnetically_shielded_biplanar_coil_design_004.png
    :class: sphx-glr-single-img





Plot difference in field


.. code-block:: default



    import seaborn as sns
    import matplotlib.pyplot as plt


    if PLOT:
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

        if SAVE_FIGURES:
            plt.savefig(SAVE_DIR + "shielding_effect.pdf")



.. image:: /auto_examples/publication_software/images/sphx_glr_pub_magnetically_shielded_biplanar_coil_design_005.png
    :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 5 minutes  46.987 seconds)


.. _sphx_glr_download_auto_examples_publication_software_pub_magnetically_shielded_biplanar_coil_design.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: pub_magnetically_shielded_biplanar_coil_design.py <pub_magnetically_shielded_biplanar_coil_design.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: pub_magnetically_shielded_biplanar_coil_design.ipynb <pub_magnetically_shielded_biplanar_coil_design.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_

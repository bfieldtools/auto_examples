.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_coil_design_minimal_eddy_current_cylindrical_coil_design.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_coil_design_minimal_eddy_current_cylindrical_coil_design.py:


Coil with minimal eddy currents
===============================
Compact example of design of a cylindrical coil surrounded by a RF shield, i.e. a conductive surface.
The effects of eddy currents due to inductive interaction with the shield is minimized


.. code-block:: default


    import numpy as np
    from mayavi import mlab
    import trimesh


    from bfieldtools.mesh_conductor import MeshConductor

    from bfieldtools.coil_optimize import optimize_streamfunctions
    from bfieldtools.contour import scalar_contour
    from bfieldtools.viz import plot_3d_current_loops, plot_data_on_vertices

    import pkg_resources

    from pyface.api import GUI

    _gui = GUI()


    # Set unit, e.g. meter or millimeter.
    # This doesn't matter, the problem is scale-invariant
    scaling_factor = 1


    # Load example coil mesh that is centered on the origin
    coilmesh = trimesh.load(
        file_obj=pkg_resources.resource_filename(
            "bfieldtools", "example_meshes/open_cylinder.stl"
        ),
        process=True,
    )

    angle = np.pi / 2
    rotation_matrix = np.array(
        [
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )

    coilmesh.apply_transform(rotation_matrix)

    coilmesh1 = coilmesh.copy()

    coilmesh2 = coilmesh.copy()


    # Create mesh class object
    coil = MeshConductor(
        verts=coilmesh1.vertices * 0.75,
        tris=coilmesh1.faces,
        fix_normals=True,
        basis_name="suh",
        N_suh=400,
    )


    def alu_sigma(T):
        ref_T = 293  # K
        ref_rho = 2.82e-8  # ohm*meter
        alpha = 0.0039  # 1/K

        rho = alpha * (T - ref_T) * ref_rho + ref_rho

        return 1 / rho


    resistivity = 1 / alu_sigma(T=293)  # room-temp Aluminium
    thickness = 0.5e-3  # 0.5 mm thick


    # Separate object for shield geometry
    shield = MeshConductor(
        verts=coilmesh2.vertices.copy() * 1.1,
        tris=coilmesh2.faces.copy(),
        fix_normals=True,
        basis_name="inner",
        resistivity=resistivity,
        thickness=thickness,
    )






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Calculating surface harmonics expansion...
    Computing the laplacian matrix...
    Computing the mass matrix...




Set up target  points and plot geometry


.. code-block:: default



    center = np.array([0, 0, 0])

    sidelength = 0.25 * scaling_factor
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
    coil.plot_mesh(figure=f, opacity=0.2)
    shield.plot_mesh(figure=f, opacity=0.2)
    mlab.points3d(*target_points.T)





.. image:: /auto_examples/coil_design/images/sphx_glr_minimal_eddy_current_cylindrical_coil_design_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <mayavi.modules.glyph.Glyph object at 0x7f0c12022e90>



Compute eddy-current coupling


.. code-block:: default


    mutual_inductance = coil.mutual_inductance(shield)

    # Take into account the field produced by currents induced into the shield
    # NB! This expression is for instantaneous step-function switching of coil current, see Eq. 18 in G.N. Peeren, 2003.

    shield.M_coupling = np.linalg.solve(-shield.inductance, mutual_inductance.T)
    secondary_C = shield.B_coupling(target_points) @ -shield.M_coupling





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Estimating 69923 MiB required for 4764 by 4764 vertices...
    Computing inductance matrix in 120 chunks (12809 MiB memory free),              when approx_far=True using more chunks is faster...
    Computing triangle-coupling matrix
    Computing the inductance matrix...
    Computing self-inductance matrix using rough quadrature (degree=2).              For higher accuracy, set quad_degree to 4 or more.
    Estimating 69923 MiB required for 4764 by 4764 vertices...
    Computing inductance matrix in 120 chunks (12745 MiB memory free),              when approx_far=True using more chunks is faster...
    Computing triangle-coupling matrix
    Inductance matrix computation took 30.77 seconds.
    Computing magnetic field coupling matrix, 4764 vertices by 672 target points... took 1.04 seconds.




Create bfield specifications used when optimizing the coil geometry


.. code-block:: default


    # The absolute target field amplitude is not of importance,
    # and it is scaled to match the C matrix in the optimization function

    target_field = np.zeros(target_points.shape)
    target_field[:, 1] = target_field[:, 1] + 1


    target_spec = {
        "coupling": coil.B_coupling(target_points),
        "abs_error": 0.01,
        "target": target_field,
    }


    from scipy.linalg import eigh

    l, U = eigh(shield.resistance, shield.inductance, eigvals=(0, 500))


    time = [0.001, 0.003, 0.005]
    eddy_error = [0.05, 0.01, 0.0025]
    # time_decay = U @ np.exp(-l[None, :]*time[:, None]) @ np.pinv(U)

    time_decay = np.zeros(
        (len(time), shield.inductance.shape[0], shield.inductance.shape[1])
    )

    induction_spec = []


    Uinv = np.linalg.pinv(U)
    for idx, t in enumerate(time):
        time_decay = U @ np.diag(np.exp(-l * t)) @ Uinv
        eddy_coupling = shield.B_coupling(target_points) @ time_decay @ shield.M_coupling
        induction_spec.append(
            {
                "coupling": eddy_coupling,
                "abs_error": eddy_error[idx],
                "rel_error": 0,
                "target": np.zeros_like(target_field),
            }
        )





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Computing magnetic field coupling matrix, 4764 vertices by 672 target points... took 1.06 seconds.
    Computing the resistance matrix...




Run QP solver to optimize stream function


.. code-block:: default


    import mosek

    coil.s, prob = optimize_streamfunctions(
        coil,
        [target_spec] + induction_spec,
        objective="minimum_inductive_energy",
        solver="MOSEK",
        solver_opts={"mosek_params": {mosek.iparam.num_threads: 8}},
    )

    from bfieldtools.mesh_conductor import StreamFunction

    shield.induced_s = StreamFunction(shield.M_coupling @ coil.s, shield)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Computing the inductance matrix...
    Computing self-inductance matrix using rough quadrature (degree=2).              For higher accuracy, set quad_degree to 4 or more.
    Estimating 69923 MiB required for 4764 by 4764 vertices...
    Computing inductance matrix in 120 chunks (11946 MiB memory free),              when approx_far=True using more chunks is faster...
    Computing triangle-coupling matrix
    Inductance matrix computation took 30.97 seconds.
    Pre-existing problem not passed, creating...
    Passing parameters to problem...
    Passing problem to solver...


    Problem
      Name                   :                 
      Objective sense        : min             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 16530           
      Cones                  : 1               
      Scalar variables       : 803             
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer started.
    Problem
      Name                   :                 
      Objective sense        : min             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 16530           
      Cones                  : 1               
      Scalar variables       : 803             
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer  - threads                : 8               
    Optimizer  - solved problem         : the dual        
    Optimizer  - Constraints            : 401
    Optimizer  - Cones                  : 1
    Optimizer  - Scalar variables       : 16530             conic                  : 402             
    Optimizer  - Semi-definite variables: 0                 scalarized             : 0               
    Factor     - setup time             : 0.29              dense det. time        : 0.00            
    Factor     - ML order time          : 0.00              GP order time          : 0.00            
    Factor     - nonzeros before factor : 8.06e+04          after factor           : 8.06e+04        
    Factor     - dense dim.             : 0                 flops                  : 1.34e+09        
    ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME  
    0   3.2e+01  1.0e+00  2.0e+00  0.00e+00   0.000000000e+00   -1.000000000e+00  1.0e+00  1.57  
    1   1.7e+01  5.2e-01  1.2e+00  -6.51e-01  8.755235291e+01   8.717745151e+01   5.2e-01  1.65  
    2   1.0e+01  3.2e-01  7.5e-01  -3.65e-01  3.120153253e+02   3.119964808e+02   3.2e-01  1.72  
    3   7.2e+00  2.2e-01  5.4e-01  -9.84e-02  6.857677666e+02   6.859575059e+02   2.2e-01  1.81  
    4   5.5e+00  1.7e-01  4.3e-01  -2.59e-01  1.163962871e+03   1.164387060e+03   1.7e-01  1.89  
    5   1.9e+00  5.8e-02  1.5e-01  -2.36e-01  7.480098289e+03   7.481027404e+03   5.8e-02  1.98  
    6   6.8e-01  2.1e-02  4.7e-02  1.04e-01   1.627369627e+04   1.627454019e+04   2.1e-02  2.05  
    7   3.5e-01  1.1e-02  1.9e-02  7.70e-01   2.132852436e+04   2.132909581e+04   1.1e-02  2.17  
    8   3.2e-01  9.8e-03  1.7e-02  5.09e-01   2.209800849e+04   2.209853837e+04   9.8e-03  2.24  
    9   1.5e-01  4.5e-03  5.9e-03  6.46e-01   2.619385304e+04   2.619417789e+04   4.5e-03  2.32  
    10  4.2e-02  1.3e-03  1.1e-03  7.82e-01   2.972842030e+04   2.972856479e+04   1.3e-03  2.45  
    11  2.2e-02  6.8e-04  4.5e-04  8.03e-01   3.075144691e+04   3.075153483e+04   6.8e-04  2.53  
    12  8.2e-04  2.5e-05  3.8e-06  8.85e-01   3.207619891e+04   3.207620363e+04   2.5e-05  2.64  
    13  1.4e-04  4.2e-06  2.6e-07  9.95e-01   3.213108619e+04   3.213108702e+04   4.2e-06  2.72  
    14  1.5e-05  6.0e-08  3.6e-09  9.99e-01   3.214223525e+04   3.214223521e+04   6.1e-09  2.79  
    15  1.6e-05  4.5e-08  2.6e-09  1.00e+00   3.214223933e+04   3.214223930e+04   4.6e-09  3.01  
    16  2.1e-05  2.3e-08  1.1e-09  1.00e+00   3.214224547e+04   3.214224545e+04   2.3e-09  3.17  
    17  1.8e-05  1.1e-08  5.1e-10  1.00e+00   3.214224855e+04   3.214224856e+04   1.2e-09  3.36  
    18  1.8e-05  1.1e-08  5.1e-10  1.00e+00   3.214224855e+04   3.214224856e+04   1.2e-09  3.59  
    19  2.7e-05  1.1e-08  9.9e-11  1.00e+00   3.214224874e+04   3.214224874e+04   1.1e-09  3.79  
    20  2.7e-05  1.1e-08  9.9e-11  1.00e+00   3.214224874e+04   3.214224874e+04   1.1e-09  4.04  
    21  2.7e-05  1.1e-08  9.9e-11  1.00e+00   3.214224874e+04   3.214224874e+04   1.1e-09  4.29  
    Optimizer terminated. Time: 4.64    


    Interior-point solution summary
      Problem status  : PRIMAL_AND_DUAL_FEASIBLE
      Solution status : OPTIMAL
      Primal.  obj: 3.2142248741e+04    nrm: 6e+04    Viol.  con: 2e-08    var: 0e+00    cones: 0e+00  
      Dual.    obj: 3.2142248738e+04    nrm: 4e+05    Viol.  con: 2e-07    var: 2e-07    cones: 0e+00  




Plot coil windings and target points


.. code-block:: default



    loops = scalar_contour(coil.mesh, coil.s.vert, N_contours=6)


    f = mlab.figure(None, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5), size=(600, 500))
    mlab.clf()

    plot_3d_current_loops(loops, colors="auto", figure=f, tube_radius=0.005)

    B_target = coil.B_coupling(target_points) @ coil.s

    mlab.quiver3d(*target_points.T, *B_target.T)

    shield.plot_mesh(
        representation="surface",
        opacity=0.5,
        cull_back=True,
        color=(0.8, 0.8, 0.8),
        figure=f,
    )
    shield.plot_mesh(
        representation="surface",
        opacity=1,
        cull_front=True,
        color=(0.8, 0.8, 0.8),
        figure=f,
    )

    f.scene.camera.parallel_projection = 1

    f.scene.camera.zoom(1.4)




.. image:: /auto_examples/coil_design/images/sphx_glr_minimal_eddy_current_cylindrical_coil_design_002.png
    :class: sphx-glr-single-img





For comparison, let's see how the coils look when we ignore the conducting shield


.. code-block:: default



    coil.unshielded_s, coil.unshielded_prob = optimize_streamfunctions(
        coil,
        [target_spec],
        objective="minimum_inductive_energy",
        solver="MOSEK",
        solver_opts={"mosek_params": {mosek.iparam.num_threads: 8}},
    )

    shield.unshielded_induced_s = StreamFunction(
        shield.M_coupling @ coil.unshielded_s, shield
    )

    loops = scalar_contour(coil.mesh, coil.unshielded_s.vert, N_contours=6)

    f = mlab.figure(None, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5), size=(600, 500))
    mlab.clf()

    plot_3d_current_loops(loops, colors="auto", figure=f, tube_radius=0.005)

    B_target_unshielded = coil.B_coupling(target_points) @ coil.unshielded_s

    mlab.quiver3d(*target_points.T, *B_target_unshielded.T)

    shield.plot_mesh(
        representation="surface",
        opacity=0.5,
        cull_back=True,
        color=(0.8, 0.8, 0.8),
        figure=f,
    )
    shield.plot_mesh(
        representation="surface",
        opacity=1,
        cull_front=True,
        color=(0.8, 0.8, 0.8),
        figure=f,
    )

    f.scene.camera.parallel_projection = 1

    f.scene.camera.zoom(1.4)





.. image:: /auto_examples/coil_design/images/sphx_glr_minimal_eddy_current_cylindrical_coil_design_003.png
    :class: sphx-glr-single-img


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
      Constraints            : 4434            
      Cones                  : 1               
      Scalar variables       : 803             
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer started.
    Problem
      Name                   :                 
      Objective sense        : min             
      Type                   : CONIC (conic optimization problem)
      Constraints            : 4434            
      Cones                  : 1               
      Scalar variables       : 803             
      Matrix variables       : 0               
      Integer variables      : 0               

    Optimizer  - threads                : 8               
    Optimizer  - solved problem         : the dual        
    Optimizer  - Constraints            : 401
    Optimizer  - Cones                  : 1
    Optimizer  - Scalar variables       : 4434              conic                  : 402             
    Optimizer  - Semi-definite variables: 0                 scalarized             : 0               
    Factor     - setup time             : 0.06              dense det. time        : 0.00            
    Factor     - ML order time          : 0.00              GP order time          : 0.00            
    Factor     - nonzeros before factor : 8.06e+04          after factor           : 8.06e+04        
    Factor     - dense dim.             : 0                 flops                  : 3.67e+08        
    ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME  
    0   3.2e+01  1.0e+00  2.0e+00  0.00e+00   0.000000000e+00   -1.000000000e+00  1.0e+00  0.38  
    1   2.5e+01  7.8e-01  2.4e-01  2.19e+00   3.606895285e+01   3.532195174e+01   7.8e-01  0.41  
    2   1.4e+00  4.2e-02  6.7e-03  1.32e+00   4.778977359e+01   4.776570562e+01   4.2e-02  0.43  
    3   9.6e-02  3.0e-03  8.7e-05  1.06e+00   4.681593779e+01   4.681405007e+01   3.0e-03  0.46  
    4   1.9e-02  5.8e-04  8.8e-06  1.00e+00   4.676836958e+01   4.676801715e+01   5.8e-04  0.49  
    5   1.7e-04  5.1e-06  8.1e-09  1.00e+00   4.677179327e+01   4.677179029e+01   5.1e-06  0.52  
    6   6.2e-06  1.9e-07  6.1e-11  1.00e+00   4.677191103e+01   4.677191091e+01   1.9e-07  0.54  
    7   1.1e-06  4.0e-08  1.2e-13  1.00e+00   4.677191628e+01   4.677191655e+01   1.9e-13  0.59  
    Optimizer terminated. Time: 0.61    


    Interior-point solution summary
      Problem status  : PRIMAL_AND_DUAL_FEASIBLE
      Solution status : OPTIMAL
      Primal.  obj: 4.6771916277e+01    nrm: 9e+01    Viol.  con: 4e-14    var: 0e+00    cones: 1e-13  
      Dual.    obj: 4.6771916550e+01    nrm: 4e+01    Viol.  con: 0e+00    var: 1e-08    cones: 0e+00  




Finally, let's compare the time-courses


.. code-block:: default



    tmin, tmax = 0, 0.025
    Fs = 2000

    time = np.linspace(tmin, tmax, int(Fs * (tmax - tmin) + 1))

    time_decay = np.zeros(
        (len(time), shield.inductance.shape[0], shield.inductance.shape[1])
    )

    Uinv = np.linalg.pinv(U)
    for idx, t in enumerate(time):
        time_decay[idx] = U @ np.diag(np.exp(-l * t)) @ Uinv


    B_t = shield.B_coupling(target_points) @ (time_decay @ shield.induced_s).T

    unshieldedB_t = (
        shield.B_coupling(target_points) @ (time_decay @ shield.unshielded_induced_s).T
    )

    import matplotlib.pyplot as plt


    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 4))
    ax.plot(
        time * 1e3,
        np.mean(np.linalg.norm(B_t, axis=1), axis=0).T,
        "k-",
        label="Minimized",
        linewidth=1.5,
    )
    ax.set_ylabel("Transient field amplitude")
    ax.semilogy(
        time * 1e3,
        np.mean(np.linalg.norm(unshieldedB_t, axis=1), axis=0).T,
        "k--",
        label="Ignored",
        linewidth=1.5,
    )
    ax.set_xlabel("Time (ms)")


    ax.set_ylim(1e-4, 0.5)
    ax.set_xlim(0, 25)


    plt.grid(which="both", axis="y", alpha=0.1)

    plt.legend()
    fig.tight_layout()

    ax.vlines([1, 5, 10, 20], 1e-4, 0.5, alpha=0.1, linewidth=3, color="r")



.. image:: /auto_examples/coil_design/images/sphx_glr_minimal_eddy_current_cylindrical_coil_design_004.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <matplotlib.collections.LineCollection object at 0x7f0c11e01510>




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 3 minutes  8.916 seconds)

**Estimated memory usage:**  9213 MB


.. _sphx_glr_download_auto_examples_coil_design_minimal_eddy_current_cylindrical_coil_design.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: minimal_eddy_current_cylindrical_coil_design.py <minimal_eddy_current_cylindrical_coil_design.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: minimal_eddy_current_cylindrical_coil_design.ipynb <minimal_eddy_current_cylindrical_coil_design.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_

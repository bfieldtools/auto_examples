.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_publication_software_mne_interpolation.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_publication_software_mne_interpolation.py:


Field interpolation example
============================


.. code-block:: default

    import numpy as np
    from bfieldtools.mesh_conductor import MeshConductor, StreamFunction
    from mayavi import mlab
    import trimesh

    import mne

    PLOT = True
    IMPORT_MNE_DATA = True

    SAVE_MNE_DATA = True
    SAVE_DIR = "."

    if IMPORT_MNE_DATA:

        from mne.datasets import sample

        data_path = sample.data_path()
        fname = data_path + "/MEG/sample/sample_audvis-ave.fif"
        # Reading
        condition = "Left Auditory"
        evoked = mne.read_evokeds(fname, condition=condition, baseline=(None, 0), proj=True)
        evoked.pick_types(meg="mag")
        evoked.plot(exclude=[], time_unit="s")

        i0, i1 = evoked.time_as_index(0.08)[0], evoked.time_as_index(0.09)[0]
        field = evoked.data[:, i0:i1].mean(axis=1)

        # Read BEM for surface geometry and transform to correct coordinate system
        import os.path as op

        subject = "sample"
        subjects_dir = op.join(data_path, "subjects")
        bem_fname = op.join(
            subjects_dir, subject, "bem", subject + "-5120-5120-5120-bem-sol.fif"
        )
        bem = mne.read_bem_solution(bem_fname)

        # Head mesh 0
        # Innerskull mesh 2
        surf_index = 2

        trans_fname = op.join(data_path, "MEG", "sample", "sample_audvis_raw-trans.fif")
        trans0 = mne.read_trans(trans_fname)
        R = trans0["trans"][:3, :3]
        t = trans0["trans"][:3, 3]
        # Surface from MRI to HEAD
        rr = (bem["surfs"][surf_index]["rr"] - t) @ R
        # Surface from HEAD to DEVICE
        trans1 = evoked.info["dev_head_t"]
        R = trans1["trans"][:3, :3]
        t = trans1["trans"][:3, 3]
        rr = (rr - t) @ R

        mesh = trimesh.Trimesh(rr, bem["surfs"][surf_index]["tris"])
        mlab.triangular_mesh(*mesh.vertices.T, mesh.faces)

        surf_index = 0

        R = trans0["trans"][:3, :3]
        t = trans0["trans"][:3, 3]
        # Surface from MRI to HEAD
        rr = (bem["surfs"][surf_index]["rr"] - t) @ R
        # Surface from HEAD to DEVICE
        R = trans1["trans"][:3, :3]
        t = trans1["trans"][:3, 3]
        rr = (rr - t) @ R
        head = trimesh.Trimesh(rr, bem["surfs"][surf_index]["tris"])

        mesh = head

        # Sensor locations and directions in DEVICE coordinate system
        p = np.array(
            [
                ch["loc"][:3]
                for ch in evoked.info["chs"]
                if ch["ch_name"][-1] == "1" and ch["ch_name"][:3] == "MEG"
            ]
        )
        n = np.array(
            [
                ch["loc"][-3:]
                for ch in evoked.info["chs"]
                if ch["ch_name"][-1] == "1" and ch["ch_name"][:3] == "MEG"
            ]
        )

        from mne.datasets import sample

        data_path = sample.data_path()
        fname = data_path + "/MEG/sample/sample_audvis-ave.fif"
        # Reading
        condition = "Left Auditory"
        evoked = mne.read_evokeds(fname, condition=condition, baseline=(None, 0), proj=True)
        evoked.pick_types(meg="mag")
        evoked.plot(exclude=[], time_unit="s")

        i0, i1 = evoked.time_as_index(0.08)[0], evoked.time_as_index(0.09)[0]
        field = evoked.data[:, i0:i1].mean(axis=1)

        # Read BEM for surface geometry and transform to correct coordinate system
        import os.path as op

        subject = "sample"
        subjects_dir = op.join(data_path, "subjects")
        bem_fname = op.join(
            subjects_dir, subject, "bem", subject + "-5120-5120-5120-bem-sol.fif"
        )
        bem = mne.read_bem_solution(bem_fname)

        # Head mesh 0
        # Innerskull mesh 2
        surf_index = 2

        trans_fname = op.join(data_path, "MEG", "sample", "sample_audvis_raw-trans.fif")
        trans0 = mne.read_trans(trans_fname)
        R = trans0["trans"][:3, :3]
        t = trans0["trans"][:3, 3]
        # Surface from MRI to HEAD
        rr = (bem["surfs"][surf_index]["rr"] - t) @ R
        # Surface from HEAD to DEVICE
        trans1 = evoked.info["dev_head_t"]
        R = trans1["trans"][:3, :3]
        t = trans1["trans"][:3, 3]
        rr = (rr - t) @ R

        mesh = trimesh.Trimesh(rr, bem["surfs"][surf_index]["tris"])
        mlab.triangular_mesh(*mesh.vertices.T, mesh.faces)

        surf_index = 0

        R = trans0["trans"][:3, :3]
        t = trans0["trans"][:3, 3]
        # Surface from MRI to HEAD
        rr = (bem["surfs"][surf_index]["rr"] - t) @ R
        # Surface from HEAD to DEVICE
        R = trans1["trans"][:3, :3]
        t = trans1["trans"][:3, 3]
        rr = (rr - t) @ R
        head = trimesh.Trimesh(rr, bem["surfs"][surf_index]["tris"])

        mesh = head

        # Sensor locations and directions in DEVICE coordinate system
        p = np.array(
            [
                ch["loc"][:3]
                for ch in evoked.info["chs"]
                if ch["ch_name"][-1] == "1" and ch["ch_name"][:3] == "MEG"
            ]
        )
        n = np.array(
            [
                ch["loc"][-3:]
                for ch in evoked.info["chs"]
                if ch["ch_name"][-1] == "1" and ch["ch_name"][:3] == "MEG"
            ]
        )

        if PLOT:
            # Plot sensor locations and directions
            mlab.triangular_mesh(
                *head.vertices.T, head.faces, color=(0.5, 0.5, 0.5), opacity=0.5
            )
            mlab.quiver3d(*p.T, *n.T, mode="arrow")

        if SAVE_MNE_DATA:
            np.savez(
                SAVE_DIR + "mne_data.npz",
                mesh=mesh,
                p=p,
                n=n,
                vertices=mesh.vertices,
                faces=mesh.faces,
            )
            evoked.save(SAVE_DIR + "left_auditory-ave.fif")


    else:

        with np.load(SAVE_DIR + "mne_data.npz", allow_pickle=True) as data:
            mesh = data["mesh"]
            p = data["p"]
            n = data["n"]
            mesh = trimesh.Trimesh(vertices=data["vertices"], faces=data["faces"])

        evoked = mne.Evoked(SAVE_DIR + "left_auditory-ave.fif")





.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/publication_software/images/sphx_glr_mne_interpolation_001.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/publication_software/images/sphx_glr_mne_interpolation_002.png
            :class: sphx-glr-multi-img

.. image:: /auto_examples/publication_software/images/sphx_glr_mne_interpolation_003.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Reading /home/rzetter/mne_data/MNE-sample-data/MEG/sample/sample_audvis-ave.fif ...
        Read a total of 4 projection items:
            PCA-v1 (1 x 102) active
            PCA-v2 (1 x 102) active
            PCA-v3 (1 x 102) active
            Average EEG reference (1 x 60) active
        Found the data of interest:
            t =    -199.80 ...     499.49 ms (Left Auditory)
            0 CTF compensation matrices available
            nave = 55 - aspect type = 100
    Projections have already been applied. Setting proj attribute to True.
    Applying baseline correction (mode: mean)
    Loading surfaces...
    Three-layer model surfaces loaded.

    Loading the solution matrix...

    Loaded linear_collocation BEM solution from /home/rzetter/mne_data/MNE-sample-data/subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif
    Reading /home/rzetter/mne_data/MNE-sample-data/MEG/sample/sample_audvis-ave.fif ...
        Read a total of 4 projection items:
            PCA-v1 (1 x 102) active
            PCA-v2 (1 x 102) active
            PCA-v3 (1 x 102) active
            Average EEG reference (1 x 60) active
        Found the data of interest:
            t =    -199.80 ...     499.49 ms (Left Auditory)
            0 CTF compensation matrices available
            nave = 55 - aspect type = 100
    Projections have already been applied. Setting proj attribute to True.
    Applying baseline correction (mode: mean)
    Loading surfaces...
    Three-layer model surfaces loaded.

    Loading the solution matrix...

    Loaded linear_collocation BEM solution from /home/rzetter/mne_data/MNE-sample-data/subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif




Fit the surface current for the auditory evoked response


.. code-block:: default



    c = MeshConductor(mesh_obj=mesh, basis_name="suh", N_suh=150)
    M = c.mass
    # B_sensors = np.sum(c.B_coupling(p) * n[:,:,None], axis=1)
    B_sensors = np.einsum("ijk,ij->ik", c.B_coupling(p), n)
    # a = np.linalg.pinv(B_sensors, rcond=1e-15) @ field
    ss = np.linalg.svd(B_sensors @ B_sensors.T, False, False)

    # reg_exps = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]
    reg_exps = [1]
    plot_this = True
    rel_errors = []
    for reg_exp in reg_exps:
        _lambda = np.max(ss) * (10 ** (-reg_exp))
        # Laplacian in the suh basis is diagonal
        BB = B_sensors.T @ B_sensors + _lambda * (-c.laplacian) / np.max(abs(c.laplacian))
        a = np.linalg.solve(BB, B_sensors.T @ field)
        # a = B_sensors.T @ np.linalg.solve(BB, field)
        s = StreamFunction(a, c)
        b_filt = B_sensors @ s

        rel_error = np.linalg.norm(b_filt - field) / np.linalg.norm(field)
        print("Relative error:", rel_error * 100, "%")
        rel_errors.append(rel_error)

        if plot_this:
            mlab.figure(bgcolor=(1, 1, 1))
            surf = s.plot(False)
            surf.actor.mapper.interpolate_scalars_before_mapping = True
            surf.module_manager.scalar_lut_manager.number_of_colors = 16




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/publication_software/images/sphx_glr_mne_interpolation_004.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/publication_software/images/sphx_glr_mne_interpolation_005.png
            :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Calculating surface harmonics expansion...
    Computing the laplacian matrix...
    Computing the mass matrix...
    Closed mesh or Neumann BC, leaving out the constant component
    Computing the mass matrix...
    Computing magnetic field coupling matrix, 2562 vertices by 102 target points... took 0.19 seconds.
    Computing the laplacian matrix...
    Relative error: 2.0549237013162216 %




%% Interpolate to the sensor surface


.. code-block:: default


    from bfieldtools.utils import load_example_mesh

    helmet = load_example_mesh("meg_helmet", process=False)
    # Bring the surface roughly to the correct place
    helmet.vertices[:, 2] -= 0.05

    # Reset coupling by hand
    c.B_coupling.reset()
    mlab.figure(bgcolor=(1, 1, 1))
    B_surf = np.sum(
        c.B_coupling(helmet.vertices) * helmet.vertex_normals[:, :, None], axis=1
    )
    # vecs = c.B_coupling(helmet.vertices)
    mlab.quiver3d(*p.T, *n.T, mode="arrow")
    scalars = B_surf @ s
    surf = mlab.triangular_mesh(
        *helmet.vertices.T, helmet.faces, scalars=scalars, colormap="seismic"
    )
    surf.actor.mapper.interpolate_scalars_before_mapping = True
    surf.module_manager.scalar_lut_manager.number_of_colors = 15
    surf2 = s.plot(False)
    surf2.actor.mapper.interpolate_scalars_before_mapping = True
    surf2.module_manager.scalar_lut_manager.number_of_colors = 15

    # mlab.figure()
    # U_surf = c.U_coupling(helmet.vertices)
    # scalars = U_surf @ s
    # surf = mlab.triangular_mesh(*helmet.vertices.T, helmet.faces, scalars=scalars,
    #                     colormap='seismic')
    # surf.actor.mapper.interpolate_scalars_before_mapping = True
    # surf.module_manager.scalar_lut_manager.number_of_colors = 15
    # surf2 = s.plot(False)
    # surf2.actor.mapper.interpolate_scalars_before_mapping = True
    # surf2.module_manager.scalar_lut_manager.number_of_colors = 15





.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/publication_software/images/sphx_glr_mne_interpolation_006.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/publication_software/images/sphx_glr_mne_interpolation_007.png
            :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Computing magnetic field coupling matrix, 2562 vertices by 2044 target points... took 1.32 seconds.




#Load simple plane mesh that is centered on the origin
 file_obj = pkg_resources.resource_filename('bfieldtools',
                'example_meshes/10x10_plane_hires.obj')
 plane = trimesh.load(file_obj, process=True)
#t = np.eye(4)
#t[1:3,1:3] = np.array([[0,1],[-1,0]])
#mesh.apply_transform(t)
 plane.vertices *= 0.03

 scalars = c.U_coupling(plane.vertices).max(axis=1)
 vert_mask = abs(scalars) > np.max(abs(scalars)/10)
 face_index = np.nonzero(plane.faces_sparse.T @ vert_mask)[0]
 plane = plane.subdivide(face_index)

 scalars = c.U_coupling(plane.vertices).max(axis=1)
 vert_mask = abs(scalars) > np.max(abs(scalars)/5)
 face_index = np.nonzero(plane.faces_sparse.T @ vert_mask)[0]
 plane = plane.subdivide(face_index)

 scalars = c.U_coupling(plane.vertices) @ s
 vert_mask = abs(scalars) > np.max(abs(scalars)/3)
 face_index = np.nonzero(plane.faces_sparse.T @ vert_mask)[0]
 plane = plane.subdivide(face_index)

 scalars = c.U_coupling(plane.vertices) @ s
 inner = abs(c.U_coupling(plane.vertices).sum(axis=1)) >1e-15
 scalars[inner] *= -1
 m = np.max(abs(scalars))/1.5
 surf1 = mlab.triangular_mesh(*plane.vertices.T, plane.faces, scalars=scalars,
                     colormap='bwr', vmin=-m, vmax=m)
 surf1.actor.mapper.interpolate_scalars_before_mapping = True
 surf1.module_manager.scalar_lut_manager.number_of_colors = 15
 surf2 = s.plot(False)
 surf2.actor.mapper.interpolate_scalars_before_mapping = True
 surf2.module_manager.scalar_lut_manager.number_of_colors = 15
#mlab.triangular_mesh(*mesh.vertices.T, mesh.faces, color=(1,1,1))

Calculate magnetic field in a box


.. code-block:: default


    Nvol = 30
    x = np.linspace(-0.125, 0.125, Nvol)
    vol_points = np.array(np.meshgrid(x, x, x, indexing="ij")).reshape(3, -1).T
    # mlab.points3d(*vol_points.T)

    c.B_coupling.reset()
    Bvol_coupling = c.B_coupling(vol_points, Nchunks=100, analytic=True)
    s = StreamFunction(a, c)
    # s = StreamFunction(a, c)
    Bvol = Bvol_coupling @ s





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Computing magnetic field coupling matrix analytically, 2562 vertices by 27000 target points... took 75.23 seconds.




Plot the computed magnetic field with streamlines


.. code-block:: default


    from bfieldtools.mesh_calculus import gradient

    # mlab.quiver3d(*vol_points.T, *Bvol.T)
    mlab.figure(bgcolor=(1, 1, 1))
    vecs = mlab.pipeline.vector_field(
        *vol_points.T.reshape(3, Nvol, Nvol, Nvol), *Bvol.T.reshape(3, Nvol, Nvol, Nvol)
    )
    vecnorm = mlab.pipeline.extract_vector_norm(vecs)

    seed_points = mesh.vertices[mesh.faces].mean(axis=1) - 0.01 * mesh.face_normals
    # c1 = MeshConductor(mesh_obj=mesh, basis_name='vertex')
    seed_vals = c.basis @ c.inductance @ s
    seed_vals_grad = np.linalg.norm(gradient(seed_vals, c.mesh), axis=0)
    mlab.triangular_mesh(
        *mesh.vertices.T, mesh.faces, scalars=abs(seed_vals) ** 2, colormap="viridis"
    )
    seed_vals = abs(seed_vals[mesh.faces].mean(axis=1)) ** 2
    seed_vals[seed_vals_grad > seed_vals_grad.max() / 1.8] = 0
    Npoints = 500
    seed_inds = np.random.choice(
        np.arange(len(seed_vals)), Npoints, False, seed_vals / seed_vals.sum()
    )
    seed_points = seed_points[seed_inds]
    # mlab.points3d(*seed_points.T, scale_factor=0.001)
    # seed_vals /= seed_vals.max()
    # rands = np.random.rand(len(seed_vals))
    # seed_points = seed_points[seed_vals > rands]

    streams = []

    for pi in seed_points:
        streamline = mlab.pipeline.streamline(
            vecnorm,
            integration_direction="both",
            colormap="BuGn",
            seed_visible=False,
            seedtype="point",
        )
        streamline.seed.widget.position = pi
        streamline.stream_tracer.terminal_speed = 3e-13
        streamline.stream_tracer.maximum_propagation = 0.1
        streamline.actor.property.render_lines_as_tubes = True
        streamline.actor.property.line_width = 4.0
        streams.append(streamline)


    # Magnetic flux
    # s2 = StreamFunction(c.inductance @ s, c)
    # mlab.figure()
    # surf2 = s2.plot(False)
    # surf2.actor.mapper.interpolate_scalars_before_mapping = True
    # surf2.module_manager.scalar_lut_manager.number_of_colors = 15

    # mlab.figure()
    # surf2 = s.plot(False)
    # surf2.actor.mapper.interpolate_scalars_before_mapping = True
    # surf2.module_manager.scalar_lut_manager.number_of_colors = 16


    # Custom colormap with alpha channel
    streamine = streams[0]
    lut = streamline.module_manager.scalar_lut_manager.lut.table.to_array()
    lut[:, -1] = np.linspace(0, 255, 256)
    streamline.module_manager.scalar_lut_manager.lut.table = lut
    streamline.module_manager.scalar_lut_manager.data_range = np.array([1.0e-13, 1.0e-12])


    ##
    for streamline in streams:
        streamline.stream_tracer.terminal_speed = 1e-13
        streamline.seed.widget.hot_spot_size = 0.1
        streamline.stream_tracer.initial_integration_step = 0.01
        streamline.stream_tracer.minimum_integration_step = 0.1

    sensors = mlab.quiver3d(*p.T, *n.T, mode="cylinder")
    sensors.glyph.glyph_source.glyph_source.height = 0.1
    sensors.actor.property.color = (0.5, 0.5, 0.5)
    sensors.actor.mapper.scalar_visibility = False
    sensors.glyph.glyph_source.glyph_source.resolution = 32
    sensors.glyph.glyph.scale_factor = 0.03
    # sensors.glyph.glyph_source.glyph_source.shaft_radius = 0.05

    grad_s = gradient(c.basis @ s, mesh, rotated=True)
    q = mlab.quiver3d(
        *(mesh.vertices[mesh.faces].mean(axis=1).T),
        *grad_s,
        colormap="viridis",
        mode="arrow"
    )

    mlab.triangular_mesh(*head.vertices.T, head.faces, color=(0.8, 0.8, 0.8), opacity=1.0)

    #
    ##    streamline.seed.widget.enabled = False
    #    streamline.actor.property.line_width = 3.0



.. image:: /auto_examples/publication_software/images/sphx_glr_mne_interpolation_008.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Computing the inductance matrix...
    Computing self-inductance matrix using rough quadrature (degree=2).              For higher accuracy, set quad_degree to 4 or more.
    Estimating 24058 MiB required for 2562 by 2562 vertices...
    Computing inductance matrix in 80 chunks (7475 MiB memory free),              when approx_far=True using more chunks is faster...
    Computing triangle-coupling matrix
    Inductance matrix computation took 10.80 seconds.

    <mayavi.modules.surface.Surface object at 0x7f50e48309b0>




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 5 minutes  59.071 seconds)


.. _sphx_glr_download_auto_examples_publication_software_mne_interpolation.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: mne_interpolation.py <mne_interpolation.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: mne_interpolation.ipynb <mne_interpolation.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_

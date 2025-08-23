"""
    def load_all_datasets(n=10000):
        DIST_CLASS_MAP = {
            "Lollipop": Lollipop,
            "SwissRoll": SwissRoll,
            "Torus": Torus,
            "Mondrian": Mondrian,
            "MultiscaleGaussian": MultiscaleGaussian,
            "VonMisesEuclidean": VonMisesEuclidean,
            "ManifoldMixture": ManifoldMixture,
            "AffineManifoldMixture": AffineManifoldMixture,
            "SquigglyManifoldMixture": SquigglyManifoldMixture,
        }
        PROJECT_ROOT = Path(r"C:\Users\User\PycharmProjects\pythonProject3")
        yaml_dirs = [PROJECT_ROOT / "dgm_geometry" / "conf" / "dataset"] + [
            PROJECT_ROOT / "dgm_geometry" / "conf" / "dataset" / "manifolds" / subdir
            for subdir in ["large", "medium", "small", "toy"]]
        def is_flow_based(dist_cfg):
            target = dist_cfg.get("_target_", "")
            if "RQNSF" in target or "Flow" in target:
                return True
            diff = dist_cfg.get("diffeomorphism_instantiator")
            if isinstance(diff, list):
                return any("RQNSF" in str(d.get("_target_", "")) or "Flow" in str(d.get("_target_", "")) for d in diff if isinstance(d, dict))
            if isinstance(diff, dict):
                return "RQNSF" in str(diff.get("_target_", "")) or "Flow" in str(diff.get("_target_", ""))
            return False
        dataset_dict = {}
        for base_dir in yaml_dirs:
            if not base_dir.exists():
                continue
            for fname in os.listdir(base_dir):
                if not fname.endswith(".yaml"):
                    continue
                full_path = base_dir / fname
                name = fname.replace(".yaml", "")
                with open(full_path, "r") as f:
                    cfg = yaml.safe_load(f)
                val_cfg = cfg.get("val")
                if not val_cfg:
                    continue
                dist_cfg = val_cfg.get("distribution") if isinstance(val_cfg.get("distribution"), dict) else cfg.get("train", {}).get("distribution")
                if not dist_cfg:
                    print(f"[Skipped] {name} missing 'distribution' config.")
                    continue
                if isinstance(dist_cfg.get("device"), dict):
                    print(f"[Skipped] {name} uses unresolved device Hydra reference.")
                    continue
                if is_flow_based(dist_cfg):
                    print(f"[Skipped] {name} uses flow-based models (e.g., RQNSF).")
                    continue
                target = dist_cfg.get("_target_", "")
                cls_name = target.split(".")[-1]
                DistClass = DIST_CLASS_MAP.get(cls_name)
                if not DistClass:
                    print(f"[Skipped] Unknown dist type in {name} ({cls_name})")
                    continue
                dist_cfg = {k: v for k, v in dist_cfg.items() if k != "_target_"}
                standardize = val_cfg.get("standardize", False)
                try:
                    for key in ["frequency", "amplitude", "scale", "kappa_control"]:
                        if isinstance(dist_cfg.get(key), str):
                            val = float(dist_cfg[key])
                            dist_cfg[key] = int(val) if val.is_integer() else val
                    if isinstance(dist_cfg.get("manifold_dims"), list):
                        dist_cfg["manifold_dims"] = [
                            int(x) if isinstance(x, str) and x.isdigit() else x
                            for x in dist_cfg["manifold_dims"]
                        ]
                    dist = DistClass(**dist_cfg)
                    dataset = LIDSyntheticDataset(size=n, distribution=dist, standardize=standardize)
                    dataset_dict[name] = {
                        "x": dataset.x.numpy(),
                        "lid": dataset.lid.numpy(),
                        "idx": dataset.idx.numpy(),
                    }
                    print(f"[Loaded] {name}: {dataset.x.shape[0]} points in {dataset.x.shape[1]}D")
                except Exception as e:
                    print(f"[Error] Failed to load {name}: {e}")
        return dataset_dict
    def get_datasets(used_params=None, n=2500):
        data_gen = skdim.datasets.BenchmarkManifolds()
        all_keys = [key for key in data_gen.dict_gen]
        keys = all_keys[0:4] + all_keys[5:13] + all_keys[14:17] + all_keys[19:21]
        # True (d, m) pairs: intrinsic and ambient dimensions
        d_vals = [10, 3, 4, 4, 2, 6, 2, 12, 20, 10, 17, 24, 2, 20, 2, 18, 24]
        m_vals = [11, 5, 6, 8, 3, 36, 3, 72, 20, 11, 18, 25, 3, 20, 3, 72, 96]
        params = [(keys[i], [d_vals[i], m_vals[i]]) for i in range(len(keys))]
        if used_params is None:
            used_params = dict(params)
        # Generate standard datasets
        pairs = [(key,
                  [data_gen.dict_gen[key](n=n, d=used_params[key][0], dim=used_params[key][1]),
                   np.repeat(used_params[key][0], n),  # true intrinsic dim
                   used_params[key][1], np.repeat(used_params[key][0], n)])  # ambient dim
                 for key in used_params]
        result = dict(pairs)
        def add_lollipop(name, func, m_val):
            data, intrinsic_dim_array = func(n)
            result[name] = [data, intrinsic_dim_array, m_val, intrinsic_dim_array]
        add_lollipop("lollipop_", lollipop_dataset, 2)
        add_lollipop("lollipop_0", lollipop_dataset_0, 3)
        add_lollipop("lollipop_0_dense_head", lollipop_dataset_0_dense_head, 3)
        #add_lollipop("plane_line_linear_half_spaces", twod_generate_touching_halfspaces, 4)
        #add_lollipop("plane_line_cube_linear_half_spaces", threed_generate_touching_halfspaces, 4)
        #add_lollipop("plane_line_plane_linear_half_spaces", twoplanes_generate_touching_halfspaces, 4)
        new_manifolds = load_all_datasets(n=n)
        new_dict = {key: [new_manifolds[key]['x'],np.array(new_manifolds[key]['lid']),new_manifolds[key]['x'].shape[1],new_manifolds[key]['idx']] for key in new_manifolds}
        result = result | new_dict
        return result
"""
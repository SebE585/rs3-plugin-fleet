class CtxAccessor:
    def get_timeline(self, ctx):
        frames = getattr(ctx, "frames", None)
        if isinstance(frames, dict) and "timeline" in frames:
            return frames["timeline"]

        for meth_name in ("get", "get_frame", "frame", "get_dataframe", "get_df"):
            meth = getattr(ctx, meth_name, None)
            if callable(meth):
                try:
                    df = meth("timeline")
                    if df is not None:
                        return df
                except Exception:
                    pass

        meta = getattr(ctx, "meta", None)
        if isinstance(meta, dict):
            frames = meta.get("frames")
            if isinstance(frames, dict):
                return frames.get("timeline")
        return None

    def set_timeline(self, ctx, df):
        frames = getattr(ctx, "frames", None)
        if isinstance(frames, dict):
            frames["timeline"] = df; return True

        for meth_name in ("set", "set_frame", "put_frame", "put"):
            meth = getattr(ctx, meth_name, None)
            if callable(meth):
                try:
                    meth("timeline", df); return True
                except Exception:
                    pass

        meta = getattr(ctx, "meta", None)
        if isinstance(meta, dict):
            frames = meta.get("frames")
            if not isinstance(frames, dict):
                frames = {}
            frames["timeline"] = df
            meta["frames"] = frames
            return True
        return False
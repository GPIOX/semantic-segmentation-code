trainer_cfg = dict(
    accelerator="cuda",
    strategy="auto",
    devices="auto",
    callbacks=None,
    fast_dev_run=False,
    max_epochs=100,
)
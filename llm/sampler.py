"""Family-proportional-to-sigma_hat sampling (the 7th method). Every SIGMA_REESTIMATE_EVERY
steps, re-estimate each family's mean group std from the recent training groups logged by
FamilyTrackingGRPOTrainer (EMA decay SIGMA_EMA_DECAY), then set the family-draw probability
proportional to sigma_hat with SIGMA_UNIFORM_MIX uniform mixing. The dataset that actually
feeds GRPOTrainer is an IterableDataset drawing from a shared, externally-mutated probability
dict, so a callback can update sampling composition mid-training without restarting the
dataloader."""
import random

from datasets import IterableDataset
from transformers import TrainerCallback

from config import SIGMA_EMA_DECAY, SIGMA_REESTIMATE_EVERY, SIGMA_UNIFORM_MIX


def _sigma_family_generator(family_rows, family_prob_state, seed):
    """GRPOTrainer requires a `datasets.Dataset`/`datasets.IterableDataset`, not an
    arbitrary torch IterableDataset -- so this is wrapped via IterableDataset.from_generator
    below rather than subclassed directly. `family_prob_state` is a mutable dict that
    SigmaSamplingCallback updates in place mid-training; because from_generator re-invokes
    this function fresh per dataloader epoch and closes over the SAME dict object (passed
    by reference through gen_kwargs), later reads see the callback's latest probabilities."""
    rng = random.Random(seed)
    families = list(family_rows.keys())
    while True:
        probs = [family_prob_state[f] for f in families]
        fam = rng.choices(families, weights=probs, k=1)[0]
        yield dict(rng.choice(family_rows[fam]))


def make_sigma_family_dataset(family_rows, family_prob_state, seed=0):
    return IterableDataset.from_generator(
        _sigma_family_generator,
        gen_kwargs=dict(family_rows=family_rows, family_prob_state=family_prob_state, seed=seed),
    )


class SigmaSamplingCallback(TrainerCallback):
    def __init__(self, trainer_ref, family_prob_state, families, reestimate_every=SIGMA_REESTIMATE_EVERY,
                 decay=SIGMA_EMA_DECAY, mix=SIGMA_UNIFORM_MIX):
        self.trainer_ref = trainer_ref  # the FamilyTrackingGRPOTrainer instance (set post-construction)
        self.state = family_prob_state
        self.families = families
        self.reestimate_every = reestimate_every
        self.decay = decay
        self.mix = mix
        self.sigma_hat = {f: 1.0 for f in families}
        self._last_processed_idx = 0
        self.history = []

    def on_step_end(self, args, state, control, **kwargs):
        log = self.trainer_ref.family_share_log
        new_entries = log[self._last_processed_idx :]
        self._last_processed_idx = len(log)
        for entry in new_entries:
            for fam in self.families:
                if fam in entry and "group_std" in entry[fam]:
                    self.sigma_hat[fam] = self.decay * self.sigma_hat[fam] + (1 - self.decay) * entry[fam]["group_std"]

        if state.global_step > 0 and state.global_step % self.reestimate_every == 0:
            total = sum(self.sigma_hat.values()) + 1e-12
            n = len(self.families)
            for fam in self.families:
                sigma_prop = self.sigma_hat[fam] / total
                self.state[fam] = self.mix * (1.0 / n) + (1 - self.mix) * sigma_prop
            self.history.append(dict(step=int(state.global_step), sigma_hat=dict(self.sigma_hat), prob=dict(self.state)))
        return control

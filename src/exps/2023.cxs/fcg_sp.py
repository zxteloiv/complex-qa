import trialbot.utils.prepend_pythonpath  # noqa
from typing import cast

from trialbot.training import TrialBot, Registry, Events, TrainingUpdater
from trialbot.training.hparamset import HyperParamSet
from trialbot.utils.root_finder import find_root

from models.fcg.inventory import Inventory
from utils.trialbot.dummy_model import DummyModel
from utils.trialbot.dummy_translator import install_dummy_translator, DummyTranslator
from utils.trialbot.setup_cli import setup as setup_cli
from shujuji import install_semantic_parsing_datasets, get_field_names_by_prefix


@Registry.hparamset()
def conf():
    p = HyperParamSet.common_settings(find_root())
    return p


def main():
    install_semantic_parsing_datasets()
    install_dummy_translator()

    args = setup_cli(seed=2021, translator='dummy', hparamset='conf', epoch=1)
    bot = TrialBot(args, 'fcg_sp', DummyModel.factory_of_init(Inventory()))
    bot.translator: DummyTranslator
    bot.translator.lazy_init(list(get_field_names_by_prefix(args.dataset)), ['nl', 'lf'])
    bot.updater = TrainingUpdater.from_bot(bot, optim_cls=lambda ps: None)

    @bot.attach_extension(Events.STARTED)
    def start_epoch(bot: TrialBot):
        inventory: Inventory = cast(Inventory, bot.model.obj)
        translator: DummyTranslator = cast(DummyTranslator, bot.translator)
        for example in bot.train_set:
            inventory.mech1(*translator.to_tuple(example))

    bot.run()
    bot.model.obj.save('cxs')


if __name__ == '__main__':
    main()

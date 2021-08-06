from __future__ import annotations

import signal
import sys
from pathlib import Path

import click
from click import UsageError

from plagdef import services
from plagdef.config import settings
from plagdef.model.models import DocumentPairMatches, Document
from plagdef.model.reporting import generate_text_report
from plagdef.repositories import DocumentFileRepository, DocumentPairMatchesJsonRepository, DocumentPairRepository


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option(prog_name='PlagDef', package_name='plagdef')
@click.argument('docdir', type=(click.Path(exists=True), bool))
@click.option('lang', '--lang', '-l', required=True, help='Bibliographical language code of the documents\' language.')
@click.option('archive_docdir', '--archive-docs', '-a', type=(click.Path(exists=True), bool),
              help='Directory containing documents which should not be compared among themselves but only with '
                   'the documents in <DOCDIR>.')
@click.option('common_docdir', '--common-docs', '-c', type=(click.Path(exists=True), bool),
              help='Directory containing documents with sentences which should be excluded from detection.')
@click.option('sim_th', '--similarity-threshold', '-s', type=click.FloatRange(0, 1), default=0.6,
              help='Similarity threshold for text matching, defaults to 0.6. Lower values may increase sensitivity, '
                   'higher ones can improve precision.')
@click.option('ocr', '--ocr', '-o', type=bool, default=True, help='Use OCR for PDFs with poor text layers.'
                                                                  'May improve text extraction but significantly '
                                                                  'reduces performance.')
@click.option('jsondir', '--json', '-j', type=click.Path(), help='Output directory for JSON reports.')
def cli(docdir: tuple[click.Path, bool], lang: str, ocr: bool, common_docdir: [click.Path, bool],
        archive_docdir: [click.Path, bool], sim_th: float, jsondir: click.Path):
    """
    \b
    PlagDef supports plagiarism detection for student assignments.
    It must be provided a <DOCDIR> <RECURSIVE> tuple and the language option.
    The chosen directory must contain at least two documents.
    For instance if you would like to recursively search <DOCDIR> the correct command looks like this:
    `plagdef <DOCDIR> True`
    """
    settings.update({'lang': lang, 'ocr': ocr, 'min_cos_sim': sim_th, 'min_dice_sim': sim_th,
                     'min_cluster_cos_sim': sim_th})
    matches = find_matches(docdir, archive_docdir, common_docdir)
    if jsondir:
        if matches:
            try:
                write_doc_pair_matches_to_json(matches, jsondir)
                click.echo(f'Successfully wrote JSON reports to {jsondir}.')
            except NotADirectoryError as e:
                raise UsageError(str(e)) from e
    else:
        text_report = generate_text_report(matches)
        click.echo(f'\n{text_report}')
    sys.exit(0)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option(prog_name='PlagDef', package_name='plagdef')
def gui():
    """
    \b
    PlagDef supports plagiarism detection for student assignments.
    The GUI for this tool is based on the Qt 6 framework and works on all platforms.
    """
    from plagdef.gui.main import MyQtApp
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = MyQtApp()
    app.window.show()
    sys.exit(app.exec())


def find_matches(docdir: tuple, archive_docdir: tuple, common_docdir: tuple) -> list[DocumentPairMatches]:
    try:
        doc_repo = DocumentFileRepository(Path(str(docdir[0])), recursive=docdir[1])
        archive_repo = common_repo = None
        if archive_docdir:
            archive_repo = DocumentFileRepository(
                Path(str(archive_docdir[0])), recursive=archive_docdir[1])
        if common_docdir:
            common_repo = DocumentFileRepository(
                Path(str(common_docdir[0])), recursive=common_docdir[1])
        settings['last_common_docdir'] = common_docdir
        return services.find_matches(doc_repo, archive_repo=archive_repo, common_doc_repo=common_repo)
    except NotADirectoryError as e:
        raise UsageError(str(e)) from e


def reanalyze_pair(doc1: Document, doc2: Document, sim: float):
    ser, last_sim = settings['ser'], settings['min_cos_sim']
    settings.update({'ser': False, 'min_cos_sim': sim, 'min_dice_sim': sim, 'min_cluster_cos_sim': sim})
    common_repo = None
    if 'last_common_docdir' in settings and settings['last_common_docdir']:
        common_repo = DocumentFileRepository(
            Path(str(settings['last_common_docdir'][0])), recursive=settings['last_common_docdir'][1])
    doc_repo = DocumentPairRepository(Document(doc1.name, doc1.path, doc1.text),
                                      Document(doc2.name, doc2.path, doc2.text))
    matches = services.find_matches(doc_repo, common_doc_repo=common_repo)
    settings.update({'ser': ser, 'min_cos_sim': last_sim, 'min_dice_sim': last_sim, 'min_cluster_cos_sim': last_sim})
    return matches


def write_doc_pair_matches_to_json(matches, jsondir):
    repo = DocumentPairMatchesJsonRepository(Path(str(jsondir)))
    services.write_json_reports(matches, repo)


def read_doc_pair_matches_from_json(jsondir) -> set[DocumentPairMatches]:
    repo = DocumentPairMatchesJsonRepository(Path(str(jsondir)))
    return repo.list()


if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1 and args[1] == 'gui':
        del args[1]
        gui()
    else:
        cli()

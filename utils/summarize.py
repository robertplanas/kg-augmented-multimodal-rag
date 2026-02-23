import tqdm
import logging

LOGGER = logging.getLogger(__name__)


def summarize_objects(texts, images, tables):
    if len(texts) > 0:
        LOGGER.info("Summarizing text chunks")
        for text_obj in tqdm.tqdm(texts):
            text_obj.summarize_text()

    else:
        LOGGER.info("No chunks to summarize")

    if len(images) > 0:
        LOGGER.info("Summarizing images")
        for image in tqdm.tqdm(images):
            image.summarize_image()

    else:
        LOGGER.info("No images to summarize")

    if len(tables) > 0:
        LOGGER.info("Summarizing tables")
        for table in tqdm.tqdm(tables):
            table.summarize_table()

    else:
        LOGGER.info("No tables to summarize")

    return texts, images, tables

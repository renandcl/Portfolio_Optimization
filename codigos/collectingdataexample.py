from playwright.async_api import async_playwright
import asyncio
import datetime
import os
import automation_config

from os import environ
from os import path

environ['geckodriver_path'] = path.join(path.expanduser('/'), 'usr','local', 'bin', 'geckodriver')
environ['data_path'] = path.join(path.expanduser('~'), 'data', 'b3')

async def Update_Selic():
    """ This automation download the consolidated Selic file from B3
        and store in the local folder. It uses a webdriver and run on screen
        background actions.
    """

    # Get the path of the local folder
    data_path = os.getenv('data_path')

    async def run(playwright):

        # Create a new browser instance
        browser = await playwright.firefox.launch(headless=True)
        
        # Create a new context
        context = await browser.new_context(base_url="https://www3.bcb.gov.br")
        
        # Define the parameters for the post request
        parameters ={
                        "optSelecionaSerie": "1178",
                        "dataInicio": "26/06/1996+",
                        "dataFim": datetime.datetime.today().strftime("%d/%m/%Y"), #Today
                        "selTipoArqDownload": "1",
                        "hdOidSeriesSelecionadas": "1178",
                        "hdPaginar": "false",
                        "bilServico": "[SGSFW2301]"
                        }

        # Create a new page
        page = await browser.new_page()

        # Execute the post request
        response = await page.request.post("https://www3.bcb.gov.br/sgspub/consultarvalores/consultarValoresSeries.do?method=consultarValores",params=parameters)

        # Get the request url of response
        await page.goto(response.url)

        # Click text=CSV and download the file
        async with page.expect_download() as download_info:
            async with page.expect_popup() as popup_info:
                await page.click("text=CSV file")
            page1 = await popup_info.value
        download = await download_info.value

        # Download the file
        await download.save_as(os.path.join(data_path,'SELIC'+'.csv'))

        # Close the popup page, context, and browser
        await page1.close()
        await context.close()
        await browser.close()

    
    async def main():
        async with async_playwright() as playwright:
            await run(playwright)

    # Check if an async loop is running
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # 'RuntimeError: There is no current event loop...'
        loop = None

    # Run the async loop
    if loop and loop.is_running():
        task = loop.create_task(main())
        await task
    else:
        asyncio.run(main())


from playwright.async_api import async_playwright
from selenium import webdriver
import asyncio
import datetime
import time
import os
import keyboard
import automation_config
from PIL import Image, ImageTk
import cv2
import pytesseract
from zipfile import ZipFile
import tkinter as tk
import json

from os import environ
from os import path

environ['geckodriver_path'] = path.join(path.expanduser('/'), 'usr','local', 'bin', 'geckodriver')
environ['tesseract_path'] = path.join(path.expanduser('/'), 'usr', 'bin', 'tesseract')
environ['data_path'] = path.join(path.expanduser('~'), 'data', 'b3')

environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

async def Update_Instruments():
    """	This automation download the consolidated Instruments file from B3
        and store in the local folder. It uses a webdriver and run on screen
        background actions, and frontend keyboard actions.
    """
    async def main():
        # Get the path of the local folder    
        data_path = os.getenv('data_path')

        # Define the Mozilla gecko driver
        geckodriver_path = os.getenv('geckodriver_path')

        # Define Firefox properties on download
        profile = webdriver.FirefoxProfile()
        profile.set_preference("browser.download.folderList", 2)
        profile.set_preference("browser.download.forbid_open_with", True)
        profile.set_preference("browser.download.dir", data_path)
        profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/csv")

        # Define the webdriver
        driver = webdriver.Firefox(firefox_profile=profile, executable_path=geckodriver_path)
        driver.get('https://arquivos.b3.com.br/Web/Consolidated')
        time.sleep(10)

        # Click on the download button
        # download = driver.find_element_by_xpath("//*[contains(text(), 'Baixar arquivo')]")
        download = driver.find_element('xpath',"//a[contains(text(), 'Download file')]")
        download.click()
        time.sleep(10)
        
        # Perform keyboard actions for download
        keyboard.press_and_release('left arrow')
        keyboard.press_and_release('enter')
        
        # Close the webdriver
        driver.close()

    
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

async def Ibovespa_Hist_Quote(initial_year=None,final_year=None):
    """ This automation download the historical quote of the Ibovespa
        and store in the local folder. It uses a webdriver and run on screen
        background actions.
    """
    
    if final_year == None:
        final_year = datetime.datetime.today().year

    if initial_year == None:
        initial_year = final_year

    # Get the path of the local folder
    data_path = os.getenv('data_path')

    async def run(playwright):
        # Open new page
        chromium = playwright.chromium
        browser = await chromium.launch(headless=True)
        page = await browser.new_page()

        # Go to https://www.b3.com.br/pt_br/market-data-e-indices/indices/indices-amplos/indice-ibovespa-ibovespa-estatisticas-historicas.htm
        await page.goto("https://www.b3.com.br/pt_br/market-data-e-indices/indices/indices-amplos/indice-ibovespa-ibovespa-estatisticas-historicas.htm")

        # Get the data
        for i in reversed(range(initial_year, final_year+1)):
            # Start waiting for the download
            async with page.expect_download() as download_info:
                # Perform the action that initiates download
                await page.frame(name="bvmf_iframe").click("text=Download (ano selecionado)")
            download = await download_info.value
            # Download the file
            await download.save_as(os.path.join(data_path,'IBOV_'+str(i)+'.csv'))

            # Select year
            if i > 1997:
                await page.frame(name="bvmf_iframe").select_option("select", str(i-1))
        # ---------------------
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

async def B3_Assets_Hist_Quote(year = None):
    """ This automation download the historical quote of the B3
        and store in the local folder. It uses a webdriver and run on screen
        background actions, and frontend actions.
    """

    if year == None:
        year = datetime.datetime.today().year
        
    # Get the path of the local folder    
    data_path = os.getenv('data_path')

    async def run(playwright):

        # Create a new browser instance
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context()

        # Open new page
        page = await context.new_page()

        # Go to https://bvmf.bmfbovespa.com.br/pt-br/cotacoes-historicas/FormConsultaValida.asp?arq=COTAHIST_A2022.ZIP
        url = "https://bvmf.bmfbovespa.com.br/pt-br/cotacoes-historicas/FormConsultaValida.asp?arq=COTAHIST_A" + str(year) + ".ZIP"
        await page.goto(url)

        # Open the list of images
        with open(os.path.join(data_path,'B3_captcha','captcha_list.json')) as f:
            captcha_list = json.load(f)

        # Get next image name
        key = str(int(list(captcha_list.keys())[-1])+1)
        image_name = 'image'+key

        # Get the Captcha image
        await page.locator("img").screenshot(path = os.path.join(data_path,'B3_captcha', image_name + '.png'))

        # Create a window to fill captcha 
        window = tk.Tk()
        window.title("B3 captcha")
        
        # Create a label to show the image
        img = ImageTk.PhotoImage(Image.open(os.path.join(data_path,'B3_captcha', image_name + '.png')))
        panel = tk.Label(window, image = img)
        
        # Commands to get the captcha value and close the window
        def callback():
            txtVar.set(usrIn.get())
            button.configure(state=tk.DISABLED)
            window.destroy()

        # Create a field to input the captcha
        txtVar = tk.StringVar(None)
        usrIn = tk.Entry(window, textvariable=txtVar)
        button = tk.Button(window, text="OK", command = lambda:callback())
        window.bind('<Return>', (lambda event: callback()))

        # Show the window
        panel.pack()
        usrIn.pack()
        button.pack()
        window.after(1, lambda: window.focus_force())
        window.mainloop()

        # Get the captcha value
        user_input = txtVar.get()
        
        # Store the new captcha on the list
        captcha_list[key] = {image_name : user_input}
        
        # Save the list
        with open(os.path.join(data_path,'B3_captcha','captcha_list.json'), 'w') as f:
            json.dump(captcha_list, f)
        
        # Image treatment to black and white
        image = cv2.imread(os.path.join(data_path,'B3_captcha',image_name + '.png'))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        se = cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
        bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
        out_gray = cv2.divide(image, bg, scale=255)
        out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1] 

        # Save the treated image
        cv2.imwrite(os.path.join(data_path,'B3_captcha',image_name + 'treat.png'), out_binary)
        
        # with Image.open(os.path.join(data_path,'B3_captcha',image_name + 'treat.png') as img:
        #     img.show()

        # OCR (Optical Character Recognition) of the image
        pytesseract.pytesseract.tesseract_cmd = os.getenv('tesseract_path')
        value = pytesseract.image_to_string(Image.open(os.path.join(data_path, 'B3_captcha', image_name + 'treat.png')))

        # Define value to fill the captcha
        print('User <' + user_input + ">, OCR <" + value.strip() + ">")
        value = user_input

        # Fill input[name="txtTexto"]
        await page.fill("input[name=\"txtTexto\"]", value.strip())

        # Click text=OK
        await page.click("text=OK")
        # assert page.url == "https://bvmf.bmfbovespa.com.br/pt-br/cotacoes-historicas/FormConsultaValidaImagem.asp"

        # Click text=Download
        async with page.expect_download() as download_info:
            async with page.expect_popup() as popup_info:
                await page.click("text=Download")
            page1 = await popup_info.value
        download = await download_info.value

        # Save download
        await download.save_as(os.path.join(data_path,'COTAHIST_A'+ str(year) +'.zip'))

        # Extract file from zip
        with ZipFile(os.path.join(data_path,'COTAHIST_A'+ str(year) +'.zip'), 'r') as zipObj:
            zipObj.extractall(data_path)
        
        # Delete zip file
        os.remove(os.path.join(data_path,'COTAHIST_A'+ str(year) +'.zip'))
        
        # Close page
        await page1.close()

        # ---------------------
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
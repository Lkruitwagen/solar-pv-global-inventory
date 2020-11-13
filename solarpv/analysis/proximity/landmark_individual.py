import os, time, logging, glob, argparse

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import sys
from PIL import Image

import pandas as pd


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

JS_DROP_FILE = """
            var target = arguments[0],
                offsetX = arguments[1],
                offsetY = arguments[2],
                document = target.ownerDocument || document,
                window = document.defaultView || window;

            var input = document.createElement('INPUT');
            input.type = 'file';
            input.onchange = function () {
              var rect = target.getBoundingClientRect(),
                  x = rect.left + (offsetX || (rect.width >> 1)),
                  y = rect.top + (offsetY || (rect.height >> 1)),
                  dataTransfer = { files: this.files };

              ['dragenter', 'dragover', 'drop'].forEach(function (name) {
                var evt = document.createEvent('MouseEvent');
                evt.initMouseEvent(name, !0, !0, window, 0, 0, 0, x, y, !1, !1, !1, !1, 0, null);
                evt.dataTransfer = dataTransfer;
                target.dispatchEvent(evt);
              });

              setTimeout(function () { document.body.removeChild(input); }, 25);
            };
            document.body.appendChild(input);
            return input;
        """

def drag_and_drop_file(drop_target, path):
    driver = drop_target.parent
    file_input = driver.execute_script(JS_DROP_FILE, drop_target, 0, 0)
    file_input.send_keys(path)


class Landmark:

    def __init__(self, virtual=False):

        self.virtual=virtual

        self.options = webdriver.ChromeOptions()
        #self.options.binary_location = '/home/lucas/Downloads/chromedriver'

        if self.virtual:
            self.options.add_argument('headless')
            self.options.add_argument('window-size=1200x600')


    def get_batches(self,json_dir,done_dir, start_idx, end_idx, batch=1):

        self.done_dir = done_dir
        gj_records = [{'idx':int(os.path.split(f)[-1].split('.')[0]),'f':f} for f in glob.glob(os.path.join(json_dir,'*.geojson'))]
        gj_records = sorted(gj_records, key=lambda rec: rec['idx'])
        done_idxs = [os.path.split(f)[-1].split('.')[0]  for f in glob.glob(os.path.join(done_dir,'*'))]
        gj_records = [r for r in gj_records if str(r['idx']) not in done_idxs]
        gj_records = [r for r in gj_records if ((r['idx']>=start_idx) and (r['idx']<end_idx))]

        batches = [gj_records[batch_idx*batch:(batch_idx+1)*batch] for batch_idx in range((len(gj_records)//batch)+1)]

        logger.info(f'Running {len(gj_records)} in {len(batches)} batches')

        for bb in batches:
            self._get_batch(bb)

    def __load_driver(self):
        # Start up driver
        self.driver = webdriver.Chrome(chrome_options=self.options)
        self.driver.maximize_window()
        #driver = webdriver.Chrome('/home/lucas/Downloads/chromedriver')
        if self.virtual:
            self.driver.maximize_window()
        self.driver.get('http://www.landmarkmap.org/map')


        # Load initial map and close popup
        WebDriverWait(self.driver,13).until(lambda d: d.find_element_by_tag_name('div.close-icon.pointer'))
        self.driver.refresh()
        WebDriverWait(self.driver,13).until(lambda d: d.find_element_by_tag_name('div.close-icon.pointer'))
        obj = self.driver.find_element_by_xpath('//*[@id="root"]/div/div[2]/div[6]/div/article/div[1]')
        logger.info(f'Implictly wait {8}')
        time.sleep(8)
        #driver.implicitly_wait(8)
        obj.click()

        # Find and click analysis button
        obj = self.driver.find_element_by_xpath('//*[@id="root"]/div/div[2]/div[1]/nav/ul/li[3]')
        time.sleep(2)
        #driver.implicitly_wait(2)
        obj.click()
        time.sleep(5)
        #driver.implicitly_wait(5)


    def _get_batch(self, batch):

        self.__load_driver()       

        for rec in batch:

            stime = time.time()

            
            fname = os.path.join(self.done_dir,f'{rec["idx"]}.csv')

            # drag and drop GeoJSON to target box
            logger.info(f'Record: {rec["idx"]}')
            drop_target = self.driver.find_element_by_xpath('//*[@id="root"]/div/div[2]/div[1]/div[2]/div[3]/div[2]/div[3]/form')
            drag_and_drop_file(drop_target, rec['f'])
            #driver.implicitly_wait(5)
            time.sleep(5)


            # click that svg feature
            #print ('clicking ft')
            try:
                obj = self.driver.find_element_by_xpath('//*[@id="USER_FEATURES_layer"]') # may need to find all and do [-1]
                obj.click()

            except Exception as e:
                print ('ERROR!', e)

                """

                try:
                    javaScript = "document.getElementById('USER_FEATURES_layer').dispatchEvent(new Event('click'));"
                    self.driver.execute_script(javaScript)

                except Exception as e2:
                    print ('Bork!',e2)
                """

                sc_name = os.path.join(self.done_dir,f'{rec["idx"]}.png')
                logger.info(f'writing png: {sc_name}')
                self.driver.save_screenshot(sc_name)
                time.sleep(3)
                continue


            # select the analysis option
            #print ('sleep 6')
            #driver.implicitly_wait(6)
            time.sleep(6)
            #print ('inin_handles')
            #print (self.driver.window_handles)
            before_handles = len(self.driver.window_handles)
            obj = self.driver.find_element_by_xpath('//*[@id="root"]/div/div[2]/div[1]/div[2]/div[3]/div[2]/div/div[2]/div[3]/select/option[6]')
            obj.click()

            # change to the new tab
            #driver.implicitly_wait(3)
            time.sleep(3)
            #print ('handles')
            #print (self.driver.window_handles)
            after_handles = len(self.driver.window_handles)

            if before_handles==after_handles:

                logger.info(f'Record: {rec["idx"]}: no results {before_handles},{after_handles}')
                pd.DataFrame(columns=['country','name','identity','recognition_status','documentation_status']).to_csv(fname)

            else:

                self.driver.switch_to.window(self.driver.window_handles[-1])
                #time.sleep(1)

                # get all the rows from the table
                rows = self.driver.find_elements_by_tag_name('div.even-table-row') + self.driver.find_elements_by_tag_name('div.even-table-row')
                #print ('rows')
                logger.info(f'Found {len(rows)} records')
                records = []

                for row in rows:
                    record = {
                        'country':row.find_element_by_tag_name('div.country').text,
                        'name':row.find_element_by_tag_name('div.name').text,
                        'identity':row.find_element_by_tag_name('div.ident').text,
                        'recognition_status':row.find_element_by_tag_name('div.offic_rec').text,
                        'documentation_status':row.find_element_by_tag_name('div.rec_status').text,
                    }
                    records.append(record)

                df = pd.DataFrame.from_records(records)
                logger.info(f'Record: {rec["idx"]}: writing to file {fname}')
                df.to_csv(fname)

                self.driver.switch_to.window(self.driver.window_handles[0])

            obj = self.driver.find_element_by_xpath('//*[@id="root"]/div/div[2]/div[1]/div[2]/div[3]/div[2]/div/div[2]/div[1]/div/div')
            obj.click()
            time.sleep(3)




            etime = time.time()
            logger.info(f'Record: {rec["idx"]}: time: {etime-stime:.2f}')


        self.driver.quit()

        


    def _get_single(self,json_path, outname):

        # Start up driver
        driver = webdriver.Chrome(self.driver_loc)
        #driver = webdriver.Chrome('/home/lucas/Downloads/chromedriver')
        driver.maximize_window()
        driver.get('http://www.landmarkmap.org/map')


        # Load initial map and close popup
        WebDriverWait(driver,10).until(lambda d: d.find_element_by_tag_name('div.close-icon.pointer'))
        obj = driver.find_element_by_xpath('//*[@id="root"]/div/div[2]/div[6]/div/article/div[1]')
        logger.info(f'Implictly wait {4}')
        time.sleep(4)
        #driver.implicitly_wait(8)
        obj.click()

        # Find and click analysis button
        obj = driver.find_element_by_xpath('//*[@id="root"]/div/div[2]/div[1]/nav/ul/li[3]')
        time.sleep(2)
        #driver.implicitly_wait(2)
        obj.click()
        time.sleep(5)
        #driver.implicitly_wait(5)


        #time.sleep(5)

        # drag and drop GeoJSON to target box
        print ('getting file')
        drop_target = driver.find_element_by_xpath('//*[@id="root"]/div/div[2]/div[1]/div[2]/div[3]/div[2]/div[3]/form')
        drag_and_drop_file(drop_target, json_path)
        #driver.implicitly_wait(5)
        time.sleep(5)


        # click that svg feature
        print ('clicking ft')
        obj = driver.find_element_by_xpath('//*[@id="USER_FEATURES_layer"]')
        obj.click()

        # select the analysis option
        print ('sleep 6')
        #driver.implicitly_wait(6)
        time.sleep(6)
        print ('inin_handles')
        print (driver.window_handles)
        obj = driver.find_element_by_xpath('//*[@id="root"]/div/div[2]/div[1]/div[2]/div[3]/div[2]/div/div[2]/div[3]/select/option[6]')
        obj.click()

        # change to the new tab
        #driver.implicitly_wait(3)
        time.sleep(3)
        print ('handles')
        print (driver.window_handles)
        driver.switch_to.window(driver.window_handles[1])
        #time.sleep(1)

        # get all the rows from the table
        rows = driver.find_elements_by_tag_name('div.even-table-row') + driver.find_elements_by_tag_name('div.even-table-row')
        print ('rows')
        print (rows)
        records = []

        for row in rows:
            record = {
                'country':row.find_element_by_tag_name('div.country').text,
                'name':row.find_element_by_tag_name('div.name').text,
                'identity':row.find_element_by_tag_name('div.ident').text,
                'recognition_status':row.find_element_by_tag_name('div.offic_rec').text,
                'documentation_status':row.find_element_by_tag_name('div.rec_status').text,
            }
            records.append(record)

        df = pd.DataFrame.from_records(records)

        driver.quit()

        print (df)

        return df
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fetch indigenous area intersections.')
    parser.add_argument('start_idx', type=int, help='an integer for the accumulator')
    parser.add_argument('end_idx', type=int, help='an integer for the accumulator')
    parser.add_argument('--batch', type=int,default=10, help='an integer for the accumulator')


    args = parser.parse_args()


    scraper = Landmark(virtual=True)
    #scraper._get_single(os.path.join(os.getcwd(),'data','landmark.json'))
    json_dir = os.path.join(os.getcwd(),'data','landmark','do_individual')
    done_dir = os.path.join(os.getcwd(),'data','landmark_results','do_individual')
    scraper.get_batches(json_dir,done_dir, args.start_idx, args.end_idx, batch=args.batch)

from driver import MainDriver

driver = MainDriver()

if __name__ == "__main__":
    driver.start()
    driver.join()
    print("Finished Updating Data")